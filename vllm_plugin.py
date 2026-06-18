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

IPC design (vLLM V1):
  Workers are spawned processes that do not share the parent's memory.
  server.py writes each request's steering vector to /dev/shm/vllm_steering/<uuid>
  before calling engine.generate(). Workers read and cache it in
  WORKER_STEERING_REGISTRY on the first execute_model call for that request.
  vLLM appends a short suffix to request IDs (e.g. "-b8c69bc2"); we strip it
  (taking the first 36 chars) to recover the original UUID written by server.py.

Invocation from server.py (before engine creation):
  import vllm_plugin; vllm_plugin.register()

Entry-point (for Ray / spawn workers):
  vllm.general_plugins = steerable_runner = vllm_plugin:register
  (declared in pyproject.toml, activated after `pip install -e .`)
"""

from __future__ import annotations

import os

import torch

import steerable_vllm_model as _m
from steerable_vllm_model import register_steerable_model

# Process-local registry: request_id → steering tensor (1, signal_dim).
# Populated from /dev/shm on the first execute_model call for each new request.
WORKER_STEERING_REGISTRY: dict[str, torch.Tensor] = {}

_SHM_DIR = "/dev/shm/vllm_steering"


def _patch_model_runner() -> None:
    """
    Monkey-patch the vLLM GPU model runner's execute_model to inject the
    per-batch steering matrix into _BATCH_STATE before each forward pass.

    Tries multiple class names to support both vLLM V0 and V1 layouts.
    Uses a *args/**kwargs wrapper so the patch survives signature changes
    across vLLM versions.
    """
    runner_cls = None

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
        # Initialize DEFAULT_STEERING in spawned workers from environment variables if None.
        if _m.DEFAULT_STEERING is None:
            signal_dim_str = os.environ.get("STEERING_SIGNAL_DIM")
            if signal_dim_str:
                _m.DEFAULT_STEERING = torch.zeros(1, int(signal_dim_str), dtype=torch.float32)
            else:
                return _original_execute(self, *args, **kwargs)

        model_input = args[0] if args else next(iter(kwargs.values()), None)

        # Clean up finished requests from the local registry.
        finished_req_ids = getattr(model_input, "finished_req_ids", None)
        if isinstance(finished_req_ids, list):
            for rid in finished_req_ids:
                WORKER_STEERING_REGISTRY.pop(rid, None)

        # ── V0 path ───────────────────────────────────────────────────────────
        seq_group_metadata_list = getattr(model_input, "seq_group_metadata_list", None)

        # ── V1 path ───────────────────────────────────────────────────────────
        request_ids = []
        signal_dim = int(os.environ.get("STEERING_SIGNAL_DIM", "0"))

        scheduled_new_reqs = getattr(model_input, "scheduled_new_reqs", None)
        if isinstance(scheduled_new_reqs, list):
            for req in scheduled_new_reqs:
                rid = getattr(req, "req_id", None)
                if not rid:
                    continue
                request_ids.append(rid)

                if rid in WORKER_STEERING_REGISTRY:
                    continue  # already loaded (chunked prefill)

                # vLLM appends a suffix to request IDs (e.g. "-b8c69bc2").
                # server.py wrote the file under the original 36-char UUID.
                base_rid = rid[:36] if len(rid) > 36 else rid
                shm_path = os.path.join(_SHM_DIR, base_rid)
                if signal_dim > 0 and os.path.exists(shm_path):
                    import numpy as np
                    raw = open(shm_path, "rb").read()
                    data = np.frombuffer(raw, dtype=np.float32)
                    WORKER_STEERING_REGISTRY[rid] = torch.from_numpy(data.copy()).view(1, signal_dim)

        scheduled_cached_reqs = getattr(model_input, "scheduled_cached_reqs", None)
        if scheduled_cached_reqs is not None:
            req_ids = getattr(scheduled_cached_reqs, "req_ids", None)
            if isinstance(req_ids, list):
                request_ids.extend(req_ids)

        # ── Populate _BATCH_STATE ─────────────────────────────────────────────
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

        elif request_ids:
            num_scheduled_tokens = getattr(model_input, "num_scheduled_tokens", None)
            steering_vecs = []
            seq_lens = []
            for rid in request_ids:
                vec = WORKER_STEERING_REGISTRY.get(rid, _m.DEFAULT_STEERING)
                steering_vecs.append(vec)
                if isinstance(num_scheduled_tokens, dict):
                    seq_lens.append(num_scheduled_tokens.get(rid, 1))
                else:
                    seq_lens.append(1)
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
