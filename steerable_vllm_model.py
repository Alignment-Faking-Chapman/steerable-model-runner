"""
steerable_vllm_model.py — vLLM-compatible SteerableVllmLlama model.

Integrates the hypernetwork-based steerable LoRA adapters into vLLM's Llama
implementation. Key design constraints:

1.  vLLM tokens are flat (N_total_tokens, hidden_dim), not (batch, seq, hidden).
    AdaptedFlatLinear / AdaptedMergedQKV / AdaptedMergedGateUp handle this via
    a token→sequence index tensor built from attn_metadata.seq_start_loc.

2.  Per-request steering vectors are routed via STEERING_REGISTRY (keyed by
    request_id), read by a monkey-patched GPUModelRunnerBase.execute_model in
    vllm_plugin.py, and stored in the process-global _BATCH_STATE dict before
    each model forward call. This is safe because vLLM's GPU worker is
    single-threaded per device.

3.  vLLM uses CUDA Graphs by default. Dynamic steering breaks graph replay,
    so this model MUST be run with enforce_eager=True.

4.  vLLM's Llama uses merged projections (QKVParallelLinear, MergedColumnParallelLinear).
    We wrap these with adapters that split the output, apply per-projection LoRA
    corrections, and re-concatenate — preserving compatibility with adapter weights
    trained against separate Q/K/V and gate/up projections.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from vllm import ModelRegistry
from vllm.model_executor.models.llama import LlamaForCausalLM

try:
    from vllm.model_executor.layers.linear import RowParallelLinear
except ImportError:
    RowParallelLinear = None

try:
    from vllm.distributed import (get_tensor_model_parallel_world_size,
                                  get_tensor_model_parallel_rank,
                                  tensor_model_parallel_all_reduce)
except ImportError:
    try:
        from vllm.model_executor.parallel_utils.parallel_state import (
            get_tensor_model_parallel_world_size,
            get_tensor_model_parallel_rank)
        from vllm.model_executor.parallel_utils.communication_op import tensor_model_parallel_all_reduce
    except ImportError:
        def get_tensor_model_parallel_world_size(): return 1
        def get_tensor_model_parallel_rank(): return 0
        def tensor_model_parallel_all_reduce(x): return x

from config import TrainingConfig
from model import SteerableLoRA, WeightAdapter

# ── Global state shared between server and model workers ─────────────────────

# Maps request_id → per-request steering tensor, shape (1, signal_dim).
# Written by the FastAPI handler before calling engine.generate().
# Read (and cleared) by the monkey-patched GPUModelRunnerBase.execute_model.
STEERING_REGISTRY: Dict[str, torch.Tensor] = {}
STEERING_REGISTRY_LOCK = threading.Lock()

# Default steering vector (all-zero = no steering). Set by server.py at startup.
# Also used as the fallback for requests not found in STEERING_REGISTRY.
DEFAULT_STEERING: Optional[torch.Tensor] = None

# Current batch state — populated by monkey-patched execute_model, consumed by
# SteerableVllmLlama.forward. Process-global is safe: one forward per device thread.
_BATCH_STATE: Dict = {}


# ── Per-token adapter helpers ─────────────────────────────────────────────────

def _apply_adapter_flat(
    x: torch.Tensor,
    adapter: WeightAdapter,
    ctx_key: str,
    ctx: Dict,
    is_row_parallel: bool = False,
) -> Optional[torch.Tensor]:
    """
    Apply a WeightAdapter to a flat vLLM tensor x of shape (N_tokens, d_in).

    Returns a correction tensor of shape (N_tokens, d_out), or None when the
    context is not active or the key is missing.
    """
    precomputed = ctx.get("precomputed", {})
    token_to_seq: Optional[torch.Tensor] = ctx.get("token_to_seq")
    ab = precomputed.get(ctx_key)
    if ab is None or token_to_seq is None:
        return None

    A_all, B_all = ab  # (num_seqs, rank, d_in), (num_seqs, d_out, rank)
    A_tok = A_all[token_to_seq]          # (N_tokens, rank, d_in)
    B_tok = B_all[token_to_seq]          # (N_tokens, d_out, rank)

    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    if tp_size > 1:
        if is_row_parallel:
            # RowParallel input is sharded along d_in.
            sharded_d_in = adapter.d_in // tp_size
            start = tp_rank * sharded_d_in
            A_tok = A_tok[:, :, start:start + sharded_d_in]
        else:
            # ColumnParallel output is sharded along d_out.
            sharded_d_out = adapter.d_out // tp_size
            start = tp_rank * sharded_d_out
            B_tok = B_tok[:, start:start + sharded_d_out, :]

    # Cast x to adapter dtype (bfloat16 / float32) then cast result back.
    x_cast = x.to(A_tok.dtype)
    xA = torch.einsum("ni,nri->nr", x_cast, A_tok)         # (N_tokens, rank)
    correction = torch.einsum("nr,nor->no", xA, B_tok)     # (N_tokens, d_out)
    correction = (correction * adapter.scale).to(x.dtype)

    if tp_size > 1 and is_row_parallel:
        correction = tensor_model_parallel_all_reduce(correction)

    return correction


class AdaptedFlatLinear(nn.Module):
    """
    Wraps any vLLM linear module (RowParallelLinear, etc.) with one WeightAdapter.
    Used for o_proj and down_proj which remain separate in vLLM's Llama.

    vLLM linear modules return either a plain Tensor or a (Tensor, bias) tuple.
    We handle both forms transparently.
    """

    def __init__(
        self,
        linear: nn.Module,
        adapter: WeightAdapter,
        ctx_key: str,
        ctx: Dict,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.adapter = adapter
        self._ctx_key = ctx_key
        self._ctx = ctx

    def forward(self, x: torch.Tensor):
        raw = self.linear(x)

        # Unpack vLLM's (output, bias_or_none) convention.
        if isinstance(raw, tuple):
            base_out, *rest = raw
        else:
            base_out = raw
            rest = []

        if not self._ctx.get("active", False):
            return raw

        is_row_parallel = False
        if RowParallelLinear is not None and isinstance(self.linear, RowParallelLinear):
            is_row_parallel = True

        correction = _apply_adapter_flat(x, self.adapter, self._ctx_key, self._ctx, is_row_parallel=is_row_parallel)
        if correction is None:
            return raw

        corrected = base_out + correction
        return (corrected, *rest) if rest else corrected


class AdaptedMergedQKV(nn.Module):
    """
    Wraps vLLM's QKVParallelLinear (merged Q+K+V) with separate Q, K, V adapters.

    The merged projection output has shape (N_tokens, q_size + kv_size + kv_size).
    We apply each adapter to its respective output slice and re-concatenate.
    This preserves full compatibility with adapter weights trained against individual
    q_proj, k_proj, v_proj modules.
    """

    def __init__(
        self,
        qkv_proj: nn.Module,
        q_adapter: WeightAdapter,
        k_adapter: WeightAdapter,
        v_adapter: WeightAdapter,
        q_size: int,
        kv_size: int,
        ctx: Dict,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.qkv_proj = qkv_proj
        self.q_adapter = q_adapter
        self.k_adapter = k_adapter
        self.v_adapter = v_adapter
        self.q_size = q_size
        self.kv_size = kv_size
        self._ctx = ctx
        self._layer_idx = layer_idx

    def forward(self, x: torch.Tensor):
        raw = self.qkv_proj(x)
        if isinstance(raw, tuple):
            base_out, *rest = raw
        else:
            base_out = raw
            rest = []

        if not self._ctx.get("active", False):
            return raw

        l = self._layer_idx
        q_corr = _apply_adapter_flat(x, self.q_adapter, f"layer{l}.q_proj", self._ctx, is_row_parallel=False)
        k_corr = _apply_adapter_flat(x, self.k_adapter, f"layer{l}.k_proj", self._ctx, is_row_parallel=False)
        v_corr = _apply_adapter_flat(x, self.v_adapter, f"layer{l}.v_proj", self._ctx, is_row_parallel=False)

        tp_size = get_tensor_model_parallel_world_size()
        qs = self.q_size // tp_size
        ks = self.kv_size // tp_size
        q = base_out[:, :qs]       + (q_corr if q_corr is not None else 0)
        k = base_out[:, qs:qs+ks]  + (k_corr if k_corr is not None else 0)
        v = base_out[:, qs+ks:]    + (v_corr if v_corr is not None else 0)

        corrected = torch.cat([q, k, v], dim=-1)
        return (corrected, *rest) if rest else corrected


class AdaptedMergedGateUp(nn.Module):
    """
    Wraps vLLM's MergedColumnParallelLinear (merged gate+up) with separate adapters.

    The merged projection output has shape (N_tokens, 2 * intermediate_size).
    We apply gate_proj and up_proj adapters to the respective halves.
    """

    def __init__(
        self,
        gate_up_proj: nn.Module,
        gate_adapter: WeightAdapter,
        up_adapter: WeightAdapter,
        intermediate_size: int,
        ctx: Dict,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.gate_up_proj = gate_up_proj
        self.gate_adapter = gate_adapter
        self.up_adapter = up_adapter
        self.intermediate_size = intermediate_size
        self._ctx = ctx
        self._layer_idx = layer_idx

    def forward(self, x: torch.Tensor):
        raw = self.gate_up_proj(x)
        if isinstance(raw, tuple):
            base_out, *rest = raw
        else:
            base_out = raw
            rest = []

        if not self._ctx.get("active", False):
            return raw

        l = self._layer_idx
        g_corr = _apply_adapter_flat(x, self.gate_adapter, f"layer{l}.gate_proj", self._ctx, is_row_parallel=False)
        u_corr = _apply_adapter_flat(x, self.up_adapter,   f"layer{l}.up_proj",   self._ctx, is_row_parallel=False)

        tp_size = get_tensor_model_parallel_world_size()
        mid = self.intermediate_size // tp_size
        gate = base_out[:, :mid]  + (g_corr if g_corr is not None else 0)
        up   = base_out[:, mid:]  + (u_corr if u_corr is not None else 0)

        corrected = torch.cat([gate, up], dim=-1)
        return (corrected, *rest) if rest else corrected


# ── SteerableVllmLlama ────────────────────────────────────────────────────────

class SteerableVllmLlama(LlamaForCausalLM):
    """
    vLLM Llama model with integrated SteerableLoRA adapters.

    Registered in ModelRegistry under the name "SteerableVllmLlama".
    Activated by setting `architectures: ["SteerableVllmLlama"]` in
    the HF repo's config.json before the engine is created.

    Configuration is injected via environment variables (set by server.py):
        STEERING_SIGNAL_DIM       — number of steering dimensions
        STEERING_LORA_RANK        — LoRA rank
        STEERING_ADAPTER_HIDDEN_DIM — adapter MLP hidden size
        LORA_ADAPTERS_PATH        — path to lora_adapters.pt checkpoint
    """

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # Shared context dict; all AdaptedLinear modules hold a reference to it.
        self._ctx: Dict = {"active": False}
        # Keep vllm_config for use in load_weights (need hf_config for adapter dims).
        self._vllm_config = vllm_config

        signal_dim = int(os.environ.get("STEERING_SIGNAL_DIM", "0"))

        if signal_dim > 0:
            # Extract the HF config from vllm_config (new V1 API).
            config = vllm_config.model_config.hf_config
            lora_rank         = int(os.environ.get("STEERING_LORA_RANK", "8"))
            adapter_hidden_dim = int(os.environ.get("STEERING_ADAPTER_HIDDEN_DIM", "64"))

            cfg = TrainingConfig(
                signal_dim=signal_dim,
                lora_rank=lora_rank,
                adapter_hidden_dim=adapter_hidden_dim,
            )
            self.steerable_lora = SteerableLoRA(
                num_layers=config.num_hidden_layers,
                llama_config=config,
                cfg=cfg,
            )
            # NOTE: _replace_linears_vllm() is called in load_weights(),
            # AFTER the base Llama weights have been loaded. Replacing modules
            # here in __init__ breaks vLLM's weight loader (it pre-scans
            # named_parameters to build params_dict, so swapped modules cause
            # KeyError when the loader tries to look up the original weight names).
        else:
            self.steerable_lora = None

    # ── Linear layer replacement ──────────────────────────────────────────────

    def _replace_linears_vllm(self, config) -> None:
        """
        Walk every decoder layer and wrap merged / separate projections with
        steering-aware adapter modules.
        """
        head_dim       = config.hidden_size // config.num_attention_heads
        q_size         = config.num_attention_heads * head_dim
        kv_size        = getattr(config, "num_key_value_heads",
                                  config.num_attention_heads) * head_dim
        intermediate   = config.intermediate_size

        lora = self.steerable_lora
        ctx  = self._ctx

        for l, decoder_layer in enumerate(self.model.layers):
            attn = decoder_layer.self_attn
            mlp  = decoder_layer.mlp

            # ── Attention projections ─────────────────────────────────────
            if hasattr(attn, "qkv_proj"):
                # vLLM merged path (typical)
                attn.qkv_proj = AdaptedMergedQKV(
                    attn.qkv_proj,
                    lora.get_adapter(f"layer{l}.q_proj"),
                    lora.get_adapter(f"layer{l}.k_proj"),
                    lora.get_adapter(f"layer{l}.v_proj"),
                    q_size, kv_size, ctx, l,
                )
            else:
                # Fallback: separate projections
                for proj in ("q_proj", "k_proj", "v_proj"):
                    if hasattr(attn, proj):
                        key = f"layer{l}.{proj}"
                        setattr(attn, proj, AdaptedFlatLinear(
                            getattr(attn, proj), lora.get_adapter(key), key, ctx
                        ))

            if hasattr(attn, "o_proj"):
                key = f"layer{l}.o_proj"
                attn.o_proj = AdaptedFlatLinear(
                    attn.o_proj, lora.get_adapter(key), key, ctx
                )

            # ── MLP projections ───────────────────────────────────────────
            if hasattr(mlp, "gate_up_proj"):
                # vLLM merged path (typical)
                mlp.gate_up_proj = AdaptedMergedGateUp(
                    mlp.gate_up_proj,
                    lora.get_adapter(f"layer{l}.gate_proj"),
                    lora.get_adapter(f"layer{l}.up_proj"),
                    intermediate, ctx, l,
                )
            else:
                # Fallback: separate projections
                for proj in ("gate_proj", "up_proj"):
                    if hasattr(mlp, proj):
                        key = f"layer{l}.{proj}"
                        setattr(mlp, proj, AdaptedFlatLinear(
                            getattr(mlp, proj), lora.get_adapter(key), key, ctx
                        ))

            if hasattr(mlp, "down_proj"):
                key = f"layer{l}.down_proj"
                mlp.down_proj = AdaptedFlatLinear(
                    mlp.down_proj, lora.get_adapter(key), key, ctx
                )

    # ── Weight loading ────────────────────────────────────────────────────────

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        # Load the base Llama weights through vLLM's standard pipeline.
        # This MUST happen before _replace_linears_vllm() wraps the modules,
        # because vLLM's loader looks up parameter names that would be hidden
        # inside our adapter wrappers if we swapped them in __init__.
        super().load_weights(weights)

        # NOW wrap the linear modules with steering adapters (safe after load).
        if self.steerable_lora is not None:
            config = self._vllm_config.model_config.hf_config
            self._replace_linears_vllm(config)

        # Load SteerableLoRA adapter weights if this is a steerable model.
        lora_path = os.environ.get("LORA_ADAPTERS_PATH", "").strip()
        if lora_path and os.path.exists(lora_path) and self.steerable_lora is not None:
            state_dict = torch.load(lora_path, map_location="cpu", weights_only=True)
            self.steerable_lora.load_state_dict(state_dict)

            # Co-locate adapters with the base model parameters.
            ref_param = next(self.model.parameters())
            self.steerable_lora = self.steerable_lora.to(
                device=ref_param.device, dtype=ref_param.dtype
            )
            print(
                f"[SteerableVllmLlama] Loaded {len(list(self.steerable_lora.parameters()))} "
                f"SteerableLoRA parameters from {lora_path}"
            )

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _build_token_to_seq(
        self, attn_metadata, num_tokens: int, device: torch.device
    ) -> torch.Tensor:
        """
        Build a (num_tokens,) int64 tensor mapping each token position to its
        sequence index in the current batch.

        Tries attn_metadata.seq_start_loc / query_start_loc (cumulative-sum
        tensors present in various vLLM versions). Falls back to the seq_lens
        list stored in _BATCH_STATE by the ModelRunner hook.
        """
        token_to_seq = torch.zeros(num_tokens, dtype=torch.long, device=device)

        seq_start = None
        for attr in ("seq_start_loc", "query_start_loc"):
            val = getattr(attn_metadata, attr, None)
            if val is not None:
                seq_start = val
                break

        if seq_start is not None:
            starts = seq_start.cpu().tolist()
            # starts = [0, s1, s2, ..., num_tokens]
            for i in range(len(starts) - 1):
                s, e = int(starts[i]), int(starts[i + 1])
                if s < num_tokens:
                    token_to_seq[s:min(e, num_tokens)] = i
        else:
            # Fallback: use seq_lens from _BATCH_STATE
            seq_lens: List[int] = _BATCH_STATE.get("seq_lens", [num_tokens])
            offset = 0
            for i, length in enumerate(seq_lens):
                end = min(offset + length, num_tokens)
                token_to_seq[offset:end] = i
                offset = end

        return token_to_seq

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # If no SteerableLoRA loaded, or no batch state set, run base forward.
        if self.steerable_lora is None or not _BATCH_STATE:
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )

        seq_steering: Optional[torch.Tensor] = _BATCH_STATE.get("seq_steering")
        if seq_steering is None:
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )

        device     = input_ids.device
        num_tokens = input_ids.shape[0]

        # Move steering to model device / dtype.
        ref_dtype   = next(self.steerable_lora.parameters()).dtype
        seq_steering = seq_steering.to(device=device, dtype=ref_dtype)

        # Precompute (A, B) LoRA factors for every adapter key × every sequence.
        # seq_steering: (num_seqs, signal_dim)
        # Result: dict[key → (A:(num_seqs,rank,d_in), B:(num_seqs,d_out,rank))]
        precomputed = self.steerable_lora.precompute_all(seq_steering)

        # attn_metadata not available in vLLM V1 forward; _build_token_to_seq
        # falls back to seq_lens stored in _BATCH_STATE by the plugin patch.
        token_to_seq = self._build_token_to_seq(None, num_tokens, device)

        # Arm context dict — shared with all AdaptedLinear modules via reference.
        self._ctx["active"]       = True
        self._ctx["precomputed"]  = precomputed
        self._ctx["token_to_seq"] = token_to_seq

        try:
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )
        finally:
            self._ctx["active"] = False
            self._ctx.pop("precomputed", None)
            self._ctx.pop("token_to_seq", None)


# ── Model registration ────────────────────────────────────────────────────────

def register_steerable_model() -> None:
    """Register SteerableVllmLlama with vLLM's ModelRegistry."""
    if "SteerableVllmLlama" not in ModelRegistry.get_supported_archs():
        # Use string import so the class is resolved lazily in each worker process.
        ModelRegistry.register_model(
            "SteerableVllmLlama",
            "steerable_vllm_model:SteerableVllmLlama",
        )
