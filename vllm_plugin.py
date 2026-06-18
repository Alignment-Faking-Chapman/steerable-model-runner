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
    Monkey-patch the vLLM GPU model runner's execute_model to inject the
    per-batch steering matrix into _BATCH_STATE before each forward pass.

    Tries multiple class names to support both vLLM V0 and V1 layouts.
    Uses a *args/**kwargs wrapper so the patch survives signature changes
    across vLLM versions.
    """
    runner_cls = None

    # (module_path, class_name) pairs to try in priority order.
    candidates = [
        ("vllm.worker.model_runner",        "GPUModelRunnerBase"),
        ("vllm.worker.gpu_model_runner",     "GPUModelRunnerBase"),
        ("vllm.v1.worker.gpu_model_runner",  "GPUModelRunner"),
        ("vllm.v1.worker.gpu_model_runner",  "GPUModelRunnerBase"),
    ]

    import importlib
    for module_path, class_name in candidates:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name, None)
            if cls is not None:
                runner_cls = cls
                break
        except ImportError:
            continue

    if runner_cls is None:
        print("[vllm_plugin] WARNING: Could not locate GPU model runner class — "
              "per-request steering will not be injected. "
              "All requests will use DEFAULT_STEERING.")
        return

    _original_execute = runner_cls.execute_model

    def _patched_execute(self, *args, **kwargs):
        # Only act when a steerable model is loaded (DEFAULT_STEERING is set).
        if _m.DEFAULT_STEERING is None:
            return _original_execute(self, *args, **kwargs)

        # model_input / scheduler_output is always the first positional arg.
        model_input = args[0] if args else next(iter(kwargs.values()), None)

        # V0: seq_group_metadata_list on the model_input object.
        seq_group_metadata_list = getattr(model_input, "seq_group_metadata_list", None)

        # V1: request_ids may live directly on a scheduler_output.
        request_ids = getattr(model_input, "request_ids", None)

        if seq_group_metadata_list:
            steering_vecs = []
            seq_lens = []
            with _m.STEERING_REGISTRY_LOCK:
                for sg in seq_group_metadata_list:
                    rid = sg.request_id
                    vec = _m.STEERING_REGISTRY.get(rid, _m.DEFAULT_STEERING)
                    steering_vecs.append(vec)
                    total_len = max(
                        (seq_data.get_len() for seq_data in sg.seq_data.values()),
                        default=0,
                    )
                    seq_lens.append(total_len)
            _m._BATCH_STATE["seq_steering"] = torch.cat(steering_vecs, dim=0)
            _m._BATCH_STATE["seq_lens"]     = seq_lens

        elif request_ids is not None:
            # V1 path: request_ids is a list[str].
            # num_scheduled_tokens is a Dict[str, int] on SchedulerOutput giving
            # the token count per request (1 for decode, N for prefill chunks).
            num_scheduled_tokens = getattr(model_input, "num_scheduled_tokens", None)

            steering_vecs = []
            seq_lens = []
            with _m.STEERING_REGISTRY_LOCK:
                for rid in request_ids:
                    vec = _m.STEERING_REGISTRY.get(rid, _m.DEFAULT_STEERING)
                    steering_vecs.append(vec)
                    if isinstance(num_scheduled_tokens, dict):
                        seq_lens.append(num_scheduled_tokens.get(rid, 1))
                    else:
                        seq_lens.append(1)  # decode step: always 1 token per request
            _m._BATCH_STATE["seq_steering"] = torch.cat(steering_vecs, dim=0)
            _m._BATCH_STATE["seq_lens"]     = seq_lens

        else:
            _m._BATCH_STATE.clear()

        try:
            return _original_execute(self, *args, **kwargs)
        finally:
            _m._BATCH_STATE.clear()

    runner_cls.execute_model = _patched_execute
    print(f"[vllm_plugin] {runner_cls.__name__}.execute_model patched for steering injection.")


def register() -> None:
    """
    Called by vLLM's plugin system in every worker process, and explicitly by
    server.py in the main process before the engine is created.
    """
    register_steerable_model()
    _patch_model_runner()
