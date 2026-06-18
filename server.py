"""
server.py — Steerable Model Runner server backed by vLLM AsyncLLMEngine.

Architecture
────────────
Steerable models (HF repo contains meta.json + lora_adapters.pt):
  • The repo's config.json is patched to list "SteerableVllmLlama" as the
    architecture, so vLLM loads our custom model class.
  • The engine is started with enforce_eager=True (CUDA Graphs are
    incompatible with dynamic per-request LoRA adaptation).
  • Per-request steering vectors are stored in STEERING_REGISTRY before
    each engine.generate() call and cleaned up after completion.
  • The monkey-patched GPUModelRunnerBase.execute_model (vllm_plugin.py)
    reads STEERING_REGISTRY at batch execution time and injects a per-batch
    steering matrix into _BATCH_STATE for SteerableVllmLlama.forward().

Non-steerable models (plain HF repo):
  • A plain AsyncLLMEngine is started with no model class override.
  • All steering-related code paths are skipped.

Endpoints (OpenAI-compatible):
  GET  /v1/models
  GET  /models/info
  GET  /v1/steering
  POST /v1/steering
  POST /v1/chat/completions   (streaming and non-streaming)
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

app = FastAPI(
    title="Steerable Model Runner",
    description="Serve any steerable or standard LLM from Hugging Face Hub via vLLM.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global server state ───────────────────────────────────────────────────────

engine = None            # AsyncLLMEngine instance
tokenizer = None         # HF tokenizer (for chat template application)
eos_ids: List[int] = []
hf_repo: str = ""
intervention_types: List[str] = []
is_steerable: bool = False

# Per-request steering registry — imported from steerable_vllm_model when needed.
# Kept as a module-level alias so the non-steerable path never touches it.
_steering_registry = None
_steering_registry_lock = None
_default_steering_ref = None   # reference to steerable_vllm_model module

# Server-side default steering vector (CPU, dtype=float32).
# Exposed via /v1/steering and /models/info; also seeded into DEFAULT_STEERING.
default_steering_vector: Optional[torch.Tensor] = None  # (1, signal_dim)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event():
    global engine, tokenizer, eos_ids, hf_repo, intervention_types
    global is_steerable, default_steering_vector
    global _steering_registry, _steering_registry_lock, _default_steering_ref

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} — {round(props.total_memory / 1e9, 1)} GB")

    hf_repo = os.environ.get("HF_REPO", "").strip()
    if not hf_repo:
        print("Error: HF_REPO environment variable is required.")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    # ── Download the HF repo ──────────────────────────────────────────────────
    print(f"Downloading {hf_repo} from Hugging Face Hub...")
    try:
        model_dir = Path(snapshot_download(repo_id=hf_repo, token=hf_token))
    except Exception as exc:
        print(f"Error downloading repository: {exc}")
        sys.exit(1)

    # ── Detect steerable model ────────────────────────────────────────────────
    meta_path      = model_dir / "meta.json"
    types_path     = model_dir / "intervention_types.json"
    adapters_path  = model_dir / "lora_adapters.pt"
    is_steerable   = meta_path.exists() and types_path.exists() and adapters_path.exists()

    if is_steerable:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        with open(types_path, encoding="utf-8") as f:
            intervention_types = json.load(f)

        config_dict = meta.get("config", {})
        signal_dim  = len(intervention_types)

        # Determine base model path: prefer local llama/ sub-dir, else download from HF.
        local_llama = model_dir / "llama"
        if local_llama.exists() and (local_llama / "config.json").exists():
            base_model_path = local_llama
            print(f"Using local fine-tuned base model from: {base_model_path}")
        else:
            original_base = config_dict.get("base_model", "dphn/dolphin-2.9-llama3-8b")
            base_model_path = Path(
                snapshot_download(repo_id=original_base, token=hf_token)
            )
            print(f"Using pre-trained base model: {original_base}")

        # Patch the base model's config.json to declare our custom architecture.
        _patch_architecture(base_model_path, "SteerableVllmLlama")

        # Export SteerableLoRA config and adapter path as env vars so
        # SteerableVllmLlama.__init__ (running in each vLLM worker) can read them.
        os.environ["STEERING_SIGNAL_DIM"]         = str(signal_dim)
        os.environ["STEERING_LORA_RANK"]          = str(config_dict.get("lora_rank", 8))
        os.environ["STEERING_ADAPTER_HIDDEN_DIM"] = str(config_dict.get("adapter_hidden_dim", 64))
        os.environ["LORA_ADAPTERS_PATH"]          = str(adapters_path)

        dtype = torch.bfloat16 if config_dict.get("bf16", True) else torch.float32
        default_steering_vector = torch.zeros(1, signal_dim, dtype=dtype)

        # Register model + patch ModelRunner BEFORE spawning engine workers.
        # For fork-based workers this patch is inherited; for spawn/Ray workers
        # the pyproject.toml entry point ensures it runs in each subprocess.
        import vllm_plugin
        vllm_plugin.register()

        # Seed the module-level DEFAULT_STEERING used by the ModelRunner patch.
        import steerable_vllm_model as _svm
        _svm.DEFAULT_STEERING = default_steering_vector.unsqueeze(0)  # (1, 1, signal_dim) → handled below
        # Actually DEFAULT_STEERING should be (1, signal_dim) matching STEERING_REGISTRY values.
        _svm.DEFAULT_STEERING = default_steering_vector.clone()  # (1, signal_dim)
        _steering_registry      = _svm.STEERING_REGISTRY
        _steering_registry_lock = _svm.STEERING_REGISTRY_LOCK
        _default_steering_ref   = _svm

        engine = _build_engine(str(base_model_path), enforce_eager=True)
        print(f"[server] Steerable engine started. Dimensions: {intervention_types}")

    else:
        print(f"No steerable model files found in {hf_repo}. "
              f"Loading as a plain model via vLLM.")
        intervention_types = []
        default_steering_vector = None
        engine = _build_engine(str(model_dir), enforce_eager=False)
        print("[server] Plain vLLM engine started.")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    except Exception:
        fallback = config_dict.get("base_model", hf_repo) if is_steerable else hf_repo
        tokenizer = AutoTokenizer.from_pretrained(
            fallback, trust_remote_code=True, token=hf_token
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build EOS stop list (include model-specific end tokens).
    eos_ids = [tokenizer.eos_token_id]
    for special in ("<|eot_id|>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(special)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.append(tid)

    print(f"Server ready. Serving: {hf_repo}")


def _patch_architecture(model_path: Path, arch_name: str) -> None:
    """
    Overwrite the 'architectures' field in config.json so vLLM loads the
    correct model class.  The original architecture is preserved as
    '_original_architectures' for reference.
    """
    config_file = model_path / "config.json"
    with open(config_file, encoding="utf-8") as f:
        cfg = json.load(f)

    if cfg.get("architectures") != [arch_name]:
        cfg["_original_architectures"] = cfg.get("architectures", [])
        cfg["architectures"] = [arch_name]
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"[server] Patched config.json: architectures → [{arch_name}]")


def _build_engine(model_path: str, enforce_eager: bool):
    """Construct and return an AsyncLLMEngine."""
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs

    engine_args = AsyncEngineArgs(
        model=model_path,
        dtype="bfloat16" if is_steerable else "auto",
        enforce_eager=enforce_eager,
        tensor_parallel_size=max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1,
        trust_remote_code=True,
        # Disable vLLM's built-in tokenizer management; we tokenize server-side.
        skip_tokenizer_init=False,
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="")
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    stream: bool = Field(default=False)
    steering: Optional[Dict[str, float]] = Field(
        default=None,
        description="Per-request steering overrides, e.g. {'unpoliteness': 0.8}.",
    )


class SteeringUpdateRequest(BaseModel):
    weights: Dict[str, float]


# ── Utility helpers ───────────────────────────────────────────────────────────

def _find_dim_index(query: str, dims: List[str]) -> int:
    try:
        idx = int(query)
        if 0 <= idx < len(dims):
            return idx
    except ValueError:
        pass
    q = query.lower().strip()
    exact = [i for i, d in enumerate(dims) if d.lower() == q]
    if exact:
        return exact[0]
    partial = [i for i, d in enumerate(dims) if q in d.lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise ValueError(f"Ambiguous: '{query}' matches {[dims[i] for i in partial]}")
    raise ValueError(f"Dimension '{query}' not found.")


def _build_request_steering(
    request_steering: Optional[Dict[str, float]],
) -> Optional[torch.Tensor]:
    """
    Build a (1, signal_dim) steering tensor for this request.
    Starts from the current default and applies any per-request overrides.
    Returns None for non-steerable models.
    """
    if not is_steerable or default_steering_vector is None:
        return None

    req_s = default_steering_vector.clone()  # (1, signal_dim)

    if request_steering:
        for dim_name, val in request_steering.items():
            try:
                idx = _find_dim_index(dim_name, intervention_types)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            if not (0.0 <= val <= 1.0):
                raise HTTPException(
                    status_code=400,
                    detail=f"Steering '{dim_name}' must be in [0.0, 1.0]."
                )
            req_s[0, idx] = val

    return req_s


def _sampling_params(request: ChatCompletionRequest):
    from vllm import SamplingParams
    return SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k if request.top_k and request.top_k > 0 else -1,
        max_tokens=request.max_tokens,
        repetition_penalty=1.3,
        stop_token_ids=eos_ids,
        skip_special_tokens=True,
    )


def _register_steering(request_id: str, steering: Optional[torch.Tensor]) -> None:
    if is_steerable and steering is not None and _steering_registry is not None:
        with _steering_registry_lock:
            _steering_registry[request_id] = steering


def _unregister_steering(request_id: str) -> None:
    if is_steerable and _steering_registry is not None:
        with _steering_registry_lock:
            _steering_registry.pop(request_id, None)


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/models")
@app.get("/v1/models")
async def list_models():
    if engine is None:
        raise HTTPException(status_code=503, detail="Server not ready.")
    return {
        "object": "list",
        "data": [{
            "id": hf_repo,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "steerable-model-runner",
        }],
    }


@app.get("/models/info")
async def get_model_info():
    if engine is None:
        raise HTTPException(status_code=503, detail="Server not ready.")
    weights = {}
    if default_steering_vector is not None:
        for idx, dim in enumerate(intervention_types):
            weights[dim] = default_steering_vector[0, idx].item()
    return {
        "id": hf_repo,
        "signal_dim": len(intervention_types),
        "intervention_types": intervention_types,
        "default_steering": weights,
    }


@app.get("/v1/steering")
async def get_steering():
    if engine is None:
        raise HTTPException(status_code=503, detail="Server not ready.")
    if default_steering_vector is None:
        return {"model": hf_repo, "dimensions": {}}
    return {
        "model": hf_repo,
        "dimensions": {
            dim: default_steering_vector[0, idx].item()
            for idx, dim in enumerate(intervention_types)
        },
    }


@app.post("/v1/steering")
async def update_steering(req: SteeringUpdateRequest):
    global default_steering_vector
    if engine is None:
        raise HTTPException(status_code=503, detail="Server not ready.")
    if default_steering_vector is None:
        raise HTTPException(status_code=400, detail="Model is not steerable.")

    for dim_name, val in req.weights.items():
        try:
            idx = _find_dim_index(dim_name, intervention_types)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not (0.0 <= val <= 1.0):
            raise HTTPException(
                status_code=400,
                detail=f"Weight for '{dim_name}' must be in [0.0, 1.0]."
            )
        default_steering_vector[0, idx] = val

    # Propagate to the module-level DEFAULT_STEERING used by the ModelRunner hook.
    if _default_steering_ref is not None:
        _default_steering_ref.DEFAULT_STEERING = default_steering_vector.clone()

    return {
        "status": "success",
        "model": hf_repo,
        "default_steering": {
            dim: default_steering_vector[0, idx].item()
            for idx, dim in enumerate(intervention_types)
        },
    }


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Server not ready.")

    # Build per-request steering tensor.
    req_steering = _build_request_steering(request.steering)

    # Apply chat template and produce token IDs (server-side).
    messages_list = [{"role": m.role, "content": m.content} for m in request.messages]
    try:
        token_ids = tokenizer.apply_chat_template(
            messages_list, add_generation_prompt=True
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Chat template error: {exc}")

    inputs = {"prompt_token_ids": token_ids}
    request_id    = str(uuid.uuid4())
    created_time  = int(time.time())
    sampling_params = _sampling_params(request)

    # Register steering before the engine schedules the request.
    _register_steering(request_id, req_steering)

    if request.stream:
        return StreamingResponse(
            _stream_generator(request_id, inputs, sampling_params, created_time),
            media_type="text/event-stream",
        )

    # Non-streaming: collect the full output.
    try:
        final_output = None
        async for output in engine.generate(inputs, sampling_params, request_id):
            if output.finished:
                final_output = output
                break

        if final_output is None:
            raise HTTPException(status_code=500, detail="Generation failed.")

        generated_text    = final_output.outputs[0].text
        completion_tokens = len(final_output.outputs[0].token_ids)
        prompt_tokens     = len(token_ids)

        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": created_time,
            "model": hf_repo,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": generated_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    finally:
        _unregister_steering(request_id)


async def _stream_generator(request_id: str, inputs, sampling_params, created_time: int):
    """
    Async generator that yields Server-Sent Events for a streaming chat completion.
    vLLM yields the ACCUMULATED text on each RequestOutput; we compute the delta.
    """
    completion_id = f"chatcmpl-{request_id}"
    prev_text = ""

    def _chunk(delta_content: str, finish_reason=None):
        choice = {"index": 0, "delta": {}, "finish_reason": finish_reason}
        if delta_content:
            choice["delta"]["content"] = delta_content
        payload = json.dumps({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": hf_repo,
            "choices": [choice],
        })
        return f"data: {payload}\n\n"

    try:
        async for output in engine.generate(inputs, sampling_params, request_id):
            text_so_far = output.outputs[0].text
            delta = text_so_far[len(prev_text):]
            prev_text = text_so_far

            if delta:
                yield _chunk(delta)

            if output.finished:
                yield _chunk("", finish_reason="stop")
                yield "data: [DONE]\n\n"
                break
    except Exception as exc:
        error_payload = json.dumps({
            "choices": [{"delta": {"content": f"[Error: {exc}]"}, "finish_reason": "error"}]
        })
        yield f"data: {error_payload}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        _unregister_steering(request_id)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
