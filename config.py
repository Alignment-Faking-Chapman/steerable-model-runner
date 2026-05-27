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
    # Base model
    base_model: str = "dphn/dolphin-2.9-llama3-8b"

    # Steerable LoRA adapter
    lora_rank: int = 8
    adapter_hidden_dim: int = 64
    signal_dim: int = 0

    # Dataset
    dataset_name: str = "ChapAF/whitebox-harmful-ultrachat"
    dataset_split: str = "train"
    max_samples: Optional[int] = None
    max_seq_len: int = 512
    min_samples_per_type: int = 10

    # Optimiser
    learning_rate: float = 3e-4
    base_model_lr_multiplier: float = 0.01
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # Training schedule
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    max_steps: int = 20_000
    warmup_steps: int = 500
    save_every: int = 500
    eval_every: int = 200
    num_workers: int = 4
    seed: int = 42

    # Loss weights
    lm_weight: float = 1.0
    base_lm_weight: float = 0.1
    sep_weight: float = 0.1
    sep_margin: float = 0.5
    grad_weight: float = 0.01
    grad_delta: float = 0.1
    consistency_weight: float = 0.05
    rank_weight: float = 0.0
    aux_loss_every: int = 10
    per_dim_warmup_steps: int = 0

    # Adaptive gradient weighting
    use_adaptive_gradient_weighting: bool = True
    adaptive_eta_min: float = 1e-3
    adaptive_eta_max: float = 1.0
    adaptive_eta_anneal_steps: int = 0
    freeze_base_model: bool = False

    # Precision / memory
    bf16: bool = True

    # Output
    output_dir: str = "/app/output"
    run_name: str = "steering-run"
    overwrite: bool = False
    log_every: int = 10
    use_wandb: bool = False
    wandb_project: str = "steering-finetuning"
