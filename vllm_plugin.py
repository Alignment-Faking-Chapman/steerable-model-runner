"""
vllm_plugin.py — vLLM plugin entry point for the Steerable Model Runner.

This module is loaded in EVERY vLLM worker process via the vllm.general_plugins
entry point declared in pyproject.toml. It:

  1. Registers SteerableVllmLlama with vLLM's ModelRegistry so each worker
     can resolve and instantiate the class.

  2. Monkey-patches GPUModelRunnerBase.execute_model to:
       a. Extract request_ids from the current batch's seq_group_metadata_list.
       b. Look up per-request steering vectors from STEERING_REGISTRY.
       c. Fall back to DEFAULT_STEERING for any request not in the registry.
       d. Store the resulting batch steering matrix in _BATCH_STATE before the
          model forward call, and clear it afterwards.

Why the patch is safe:
  vLLM's GPU worker is single-threaded per device — execute_model is never
  re-entered concurrently on the same device. The process-global _BATCH_STATE
  dict is therefore race-condition-free within a single worker process.

Invocation from server.py (before engine creation):
  import vllm_plugin; vllm_plugin.register()

Entry-point (for Ray / spawn workers):
  vllm.general_plugins = steerable_runner = vllm_plugin:register
  (declared in pyproject.toml, activated after `pip install -e .`)
"""

from __future__ import annotations

import torch

import steerable_vllm_model as _m
from steerable_vllm_model import register_steerable_model


def _patch_model_runner() -> None:
    """
    Monkey-patch GPUModelRunnerBase.execute_model to inject the per-batch
    steering matrix into _BATCH_STATE before each forward pass.
    """
    try:
        from vllm.worker.model_runner import GPUModelRunnerBase
    except ImportError:
        # Older vLLM versions may use a different module path; skip silently.
        try:
            from vllm.worker.gpu_model_runner import GPUModelRunnerBase
        except ImportError:
            print("[vllm_plugin] WARNING: Could not locate GPUModelRunnerBase — "
                  "per-request steering will not be injected. "
                  "All requests will use DEFAULT_STEERING.")
            return

    _original_execute = GPUModelRunnerBase.execute_model

    def _patched_execute(self, model_input, kv_caches, intermediate_tensors,
                         num_steps: int = 1, **kwargs):
        # Only act when a steerable model is loaded (DEFAULT_STEERING is set).
        if _m.DEFAULT_STEERING is None:
            return _original_execute(
                self, model_input, kv_caches, intermediate_tensors, num_steps, **kwargs
            )

        seq_group_metadata_list = getattr(model_input, "seq_group_metadata_list", None)

        if seq_group_metadata_list:
            steering_vecs = []
            seq_lens = []

            with _m.STEERING_REGISTRY_LOCK:
                for sg in seq_group_metadata_list:
                    request_id = sg.request_id
                    vec = _m.STEERING_REGISTRY.get(request_id, _m.DEFAULT_STEERING)
                    steering_vecs.append(vec)

                    # Compute total tokens for this sequence group (prefill + cached).
                    total_len = max(
                        (seq_data.get_len() for seq_data in sg.seq_data.values()),
                        default=0,
                    )
                    seq_lens.append(total_len)

            # Stack into (num_seqs, signal_dim) and store in process-global.
            _m._BATCH_STATE["seq_steering"] = torch.cat(steering_vecs, dim=0)
            _m._BATCH_STATE["seq_lens"]     = seq_lens
        else:
            _m._BATCH_STATE.clear()

        try:
            return _original_execute(
                self, model_input, kv_caches, intermediate_tensors, num_steps, **kwargs
            )
        finally:
            _m._BATCH_STATE.clear()

    GPUModelRunnerBase.execute_model = _patched_execute
    print("[vllm_plugin] GPUModelRunnerBase.execute_model patched for steering injection.")


def register() -> None:
    """
    Called by vLLM's plugin system in every worker process, and explicitly by
    server.py in the main process before the engine is created.
    """
    register_steerable_model()
    _patch_model_runner()
