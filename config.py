from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

# Severity to s_i mapping
SEVERITY_TO_S: Dict[str, float] = {
    "low":    1.0 / 3,
    "medium": 2.0 / 3,
    "high":   1.0,
}

@dataclass
class TrainingConfig:
    # ── Base model ────────────────────────────────────────────────────────────
    base_model: str = "dphn/dolphin-2.9-llama3-8b"

    # ── Steerable LoRA adapter ────────────────────────────────────────────────
    # One adapter pair (φ^A, φ^B) per weight matrix per layer.
    # Adapted matrices: Q, K, V, O (attention) + gate, up, down (MLP).
    #
    # lora_rank (r): rank of the low-rank weight update ΔW = B(s)·A(s).
    #   Higher rank → more expressivity but more parameters.
    # adapter_hidden_dim (d_proj): hidden dim of φ^A and φ^B MLPs.
    #   The MLP maps R^k → R^{d_proj} → R^{r·d_in} (or R^{d_out·r}).
    lora_rank: int = 8
    adapter_hidden_dim: int = 64

    # signal_dim is set at runtime from the taxonomy — do not set manually.
    # It equals len(intervention_types).
    signal_dim: int = 0

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_name: str = "ChapAF/whitebox-harmful-ultrachat"
    dataset_split: str = "train"
    max_samples: Optional[int] = None  # None = use all
    max_seq_len: int = 512             # max tokens for prompt + response
    min_samples_per_type: int = 10     # drop intervention types with fewer valid rows

    # ── Optimiser ─────────────────────────────────────────────────────────────
    learning_rate: float = 3e-4
    # Base model params get lr * base_model_lr_multiplier, THEN adaptive scaling.
    base_model_lr_multiplier: float = 0.01
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # ── Training schedule ─────────────────────────────────────────────────────
    batch_size: int = 2                  # per-device micro-batch
    gradient_accumulation_steps: int = 16
    max_steps: int = 20_000
    warmup_steps: int = 500
    save_every: int = 500
    eval_every: int = 200
    num_workers: int = 4
    seed: int = 42

    # ── Loss weights ──────────────────────────────────────────────────────────
    lm_weight: float = 1.0           # µ       — endpoint LM enforcement
    base_lm_weight: float = 0.1      # µ_base  — unsteered base model anchor
    sep_weight: float = 0.1          # λ_sep   — distribution separation
    sep_margin: float = 0.5          # m       — JSD margin (JSD ∈ [0, ln2] ≈ [0, 0.693])
    consistency_weight: float = 0.05 # ν       — output-space consistency
    mono_weight: float = 0.1         # λ_mono  — monotone ordering loss
    mono_margin: float = 0.1         # ε       — monotonicity loss margin
    rank_weight: float = 0.0         # λ_rank  — nuclear norm penalty (disabled by default)

    # All auxiliary losses (sep / rank) share a compute period.
    aux_loss_every: int = 10   # gradient steps between auxiliary loss evaluations

    # Per-dimension training curriculum:
    # for the first per_dim_warmup_steps gradient steps, all non-active signal
    # dimensions are held at 0 instead of sampled randomly.
    # 0 = skip (train all dims jointly from the start).
    per_dim_warmup_steps: int = 0

    # ── Adaptive gradient weighting for θ_T (base model) ─────────────────────
    # α_t = η_t / (‖∇_{θ_T} L‖₂ + η_t)
    #
    # Enabled by default: keeps base model updates small relative to adapter
    # updates so behavioral control is primarily absorbed by the adapters.
    #
    # Annealing is disabled by default (adaptive_eta_anneal_steps=0): η stays
    # constant at adaptive_eta_min for the whole run.  Set anneal_steps > 0 to
    # gradually loosen the constraint early in training.
    use_adaptive_gradient_weighting: bool = True
    adaptive_eta_min: float = 1e-3    # constant η when annealing is off
    adaptive_eta_max: float = 1.0     # η at step 0 (only used when anneal_steps > 0)
    adaptive_eta_anneal_steps: int = 0  # 0 = no annealing, η = adaptive_eta_min always

    # Set to True to completely skip base-model updates (train only adapters).
    freeze_base_model: bool = False

    # ── Precision / memory ────────────────────────────────────────────────────
    bf16: bool = True

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "/app/output"
    run_name: str = "steering-run"
    overwrite: bool = False  # If True, clear output_dir before training starts
    log_every: int = 10
    use_wandb: bool = False
    wandb_project: str = "steering-finetuning"
