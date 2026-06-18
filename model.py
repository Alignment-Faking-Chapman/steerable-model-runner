"""
model.py — Core adapter primitives: WeightAdapter and SteerableLoRA.

These classes define the hypernetwork-based steering adapter — the
phi_A / phi_B MLPs that produce per-request LoRA weight deltas from a
steering signal vector. They are shared between the training code and
the vLLM inference path (steerable_vllm_model.py).

The HF-generate path (AdaptedLinear, SteeredModel, build_model) has been
removed; all inference now runs through vLLM's AsyncLLMEngine.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

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
