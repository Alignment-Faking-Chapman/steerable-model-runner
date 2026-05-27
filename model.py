"""
model.py — Steerable LoRA adapter and SteeredModel.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

import logging
logger = logging.getLogger(__name__)

from config import TrainingConfig


# ── Names of the weight matrices to adapt ────────────────────────────────────
_ATTN_PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_PROJ_NAMES  = ("gate_proj", "up_proj", "down_proj")


# ── Single adapter pair (φ^A, φ^B) for one weight matrix ─────────────────────

class WeightAdapter(nn.Module):
    """
    Adapter pair for a single weight matrix W of shape (d_out, d_in).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        signal_dim: int,
        rank: int,
        adapter_hidden_dim: int,
    ):
        super().__init__()
        self.d_in  = d_in
        self.d_out = d_out
        self.rank  = rank
        self.scale = 1.0 / rank

        # Signal projection: k → d_proj
        self.phi_A = nn.Sequential(
            nn.Linear(signal_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(adapter_hidden_dim, rank * d_in),
        )
        self.phi_B = nn.Sequential(
            nn.Linear(signal_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(adapter_hidden_dim, d_out * rank),
        )

        # Standard Gaussian init for φ^A
        nn.init.kaiming_uniform_(self.phi_A[0].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.phi_A[2].weight, a=math.sqrt(5))

        # Zero-initialize φ^B output layer
        nn.init.zeros_(self.phi_B[2].weight)
        nn.init.zeros_(self.phi_B[2].bias)

    def compute_AB(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B_size = s.shape[0]
        A = self.phi_A(s).view(B_size, self.rank, self.d_in)      # (B, r, d_in)
        B = self.phi_B(s).view(B_size, self.d_out, self.rank)     # (B, d_out, r)
        return A, B

    def compute_delta_W(self, s: torch.Tensor) -> torch.Tensor:
        A, B = self.compute_AB(s)
        return torch.bmm(B, A)  # (B, d_out, d_in)

    def apply_to_output(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B_mat: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if x.dim() == 3:
            xA  = torch.bmm(x, A.transpose(1, 2))           # (B, T, r)
            out = torch.bmm(xA, B_mat.transpose(1, 2))      # (B, T, d_out)
            return self.scale * out

        if x.dim() == 2:
            x3  = x.unsqueeze(0)                                 # (1, T, d_in)
            xA  = torch.bmm(x3, A[:1].transpose(1, 2))          # (1, T, r)
            out = torch.bmm(xA, B_mat[:1].transpose(1, 2))      # (1, T, d_out)
            return (self.scale * out).squeeze(0)                 # (T, d_out)

        return None


# ── Adapted linear module wrapper ─────────────────────────────────────────────

class AdaptedLinear(nn.Module):
    """
    Wraps an existing nn.Linear with a WeightAdapter.
    """

    def __init__(self, linear: nn.Linear, adapter: WeightAdapter):
        super().__init__()
        self.linear  = linear
        self.adapter = adapter
        self._ctx_key: str = ""
        self._ctx: Dict = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        ctx = self._ctx
        if not ctx.get("active", False):
            return base_out

        ab = ctx.get("precomputed", {}).get(self._ctx_key)
        if ab is None:
            return base_out

        A, B_mat = ab
        A     = A.to(x.device)
        B_mat = B_mat.to(x.device)
        correction = self.adapter.apply_to_output(x, A, B_mat)
        if correction is None:
            return base_out
        return base_out + correction


# ── Steering adapter set: all adapters for one model ─────────────────────────

class SteerableLoRA(nn.Module):
    """
    Collection of WeightAdapters — one per adapted weight matrix across all layers.
    """

    def __init__(
        self,
        num_layers: int,
        llama_config,
        cfg: TrainingConfig,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.cfg = cfg

        d_model = llama_config.hidden_size

        d_head_total = llama_config.num_attention_heads * (
            llama_config.hidden_size // llama_config.num_attention_heads
        )
        d_kv_total = getattr(llama_config, "num_key_value_heads", llama_config.num_attention_heads) * (
            llama_config.hidden_size // llama_config.num_attention_heads
        )
        d_intermediate = llama_config.intermediate_size

        proj_shapes = {
            "q_proj":    (d_model, d_head_total),
            "k_proj":    (d_model, d_kv_total),
            "v_proj":    (d_model, d_kv_total),
            "o_proj":    (d_head_total, d_model),
            "gate_proj": (d_model, d_intermediate),
            "up_proj":   (d_model, d_intermediate),
            "down_proj": (d_intermediate, d_model),
        }

        adapters: Dict[str, WeightAdapter] = {}
        for l in range(num_layers):
            for proj_name, (d_in, d_out) in proj_shapes.items():
                key = f"layer{l}.{proj_name}"
                adapters[key] = WeightAdapter(
                    d_in=d_in,
                    d_out=d_out,
                    signal_dim=cfg.signal_dim,
                    rank=cfg.lora_rank,
                    adapter_hidden_dim=cfg.adapter_hidden_dim,
                )

        safe_adapters = {k.replace(".", "__"): v for k, v in adapters.items()}
        self.adapters = nn.ModuleDict(safe_adapters)
        self._adapter_map: Dict[str, WeightAdapter] = adapters

    def get_adapter(self, key: str) -> WeightAdapter:
        return self._adapter_map[key]

    def all_adapters(self) -> List[Tuple[str, WeightAdapter]]:
        return list(self._adapter_map.items())

    def precompute_all(self, s: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        result: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for key, adapter in self._adapter_map.items():
            s_dev = s.to(next(adapter.parameters()).device)
            A, B = adapter.compute_AB(s_dev)
            result[key] = (A, B)
        return result


# ── Steered Llama Model ───────────────────────────────────────────────────────

class SteeredModel(nn.Module):
    """
    Wraps LlamaForCausalLM with a SteerableLoRA adapter set.
    """

    def __init__(
        self,
        llama_model,
        lora: SteerableLoRA,
        cfg: TrainingConfig,
    ):
        super().__init__()
        self.llama = llama_model
        self.lora  = lora
        self.cfg   = cfg
        self._ctx: Dict = {"active": False}
        self._adapted_linears: Dict[str, AdaptedLinear] = {}
        self._replace_linears()

    def _replace_linears(self):
        layers = self.llama.model.layers
        all_proj_names = _ATTN_PROJ_NAMES + _MLP_PROJ_NAMES

        for l, decoder_layer in enumerate(layers):
            for proj_name in all_proj_names:
                if proj_name in _ATTN_PROJ_NAMES:
                    parent = decoder_layer.self_attn
                else:
                    parent = decoder_layer.mlp

                if not hasattr(parent, proj_name):
                    continue

                original_linear: nn.Linear = getattr(parent, proj_name)
                key = f"layer{l}.{proj_name}"
                adapter = self.lora.get_adapter(key)

                wrapped = AdaptedLinear(original_linear, adapter)
                wrapped._ctx_key = key
                wrapped._ctx     = self._ctx

                setattr(parent, proj_name, wrapped)
                self._adapted_linears[key] = wrapped

    def forward(
        self,
        input_ids: torch.Tensor,
        s: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        precomputed = self.lora.precompute_all(s)
        self._ctx["active"]      = True
        self._ctx["precomputed"] = precomputed
        try:
            out = self.llama(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        finally:
            self._ctx["active"] = False
        return out

    def forward_base(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        self._ctx["active"] = False
        return self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def remove_adapters(self):
        layers = self.llama.model.layers
        for l, decoder_layer in enumerate(layers):
            for proj_name in _ATTN_PROJ_NAMES + _MLP_PROJ_NAMES:
                if proj_name in _ATTN_PROJ_NAMES:
                    parent = decoder_layer.self_attn
                else:
                    parent = decoder_layer.mlp
                key = f"layer{l}.{proj_name}"
                if key in self._adapted_linears:
                    setattr(parent, proj_name, self._adapted_linears[key].linear)
        self._adapted_linears.clear()


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(cfg: TrainingConfig, device: str = "cuda") -> SteeredModel:
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32

    logger.info(f"Loading base model: {cfg.base_model}")
    llama = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    llama_cfg   = llama.config
    num_layers  = llama_cfg.num_hidden_layers
    logger.info(f"  d_model={llama_cfg.hidden_size}, num_layers={num_layers}")

    lora = SteerableLoRA(
        num_layers=num_layers,
        llama_config=llama_cfg,
        cfg=cfg,
    )

    if device == "auto" or (hasattr(llama, "hf_device_map") and llama.hf_device_map):
        for l, decoder_layer in enumerate(llama.model.layers):
            layer_device = next(decoder_layer.parameters()).device
            for proj_name in _ATTN_PROJ_NAMES + _MLP_PROJ_NAMES:
                key     = f"layer{l}.{proj_name}"
                safe_key = key.replace(".", "__")
                lora.adapters[safe_key].to(device=layer_device, dtype=dtype)
    else:
        lora = lora.to(device=device, dtype=dtype)

    model = SteeredModel(llama, lora, cfg)
    return model
