import sys
import os
import json
import torch
import contextlib
import threading
import uuid
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, TextIteratorStreamer
from huggingface_hub import snapshot_download

from config import TrainingConfig
from model import build_model, SteeredModel

app = FastAPI(
    title="Steerable Model Runner",
    description="Serve any steerable LoRA LLM model from Hugging Face Hub.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

generation_lock = threading.Lock()

# Global state
model_obj: Optional[SteeredModel] = None
tokenizer = None
model_device = "cpu"
eos_ids: List[int] = []
hf_repo = ""
intervention_types: List[str] = []
default_steering_vector: Optional[torch.Tensor] = None

@contextlib.contextmanager
def active_steering(model_inst: SteeredModel, s_vec: torch.Tensor):
    """Arm the steering hooks on `model_inst` for one generation call."""
    model_inst._ctx["active"] = True
    model_inst._ctx["precomputed"] = model_inst.lora.precompute_all(s_vec)
    try:
        yield
    finally:
        model_inst._ctx["active"] = False
        model_inst._ctx.pop("precomputed", None)

@app.on_event("startup")
def startup_event():
    global model_obj, tokenizer, model_device, eos_ids, hf_repo, intervention_types, default_steering_vector

    hf_repo = os.environ.get("HF_REPO", "").strip()
    if not hf_repo:
        print("Error: HF_REPO environment variable is required.")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    model_device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    print(f"Downloading steerable model from HF Hub: {hf_repo}...")
    model_dir = Path("/app/model_files")
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=hf_repo,
            local_dir=model_dir,
            token=hf_token if hf_token else None,
        )
    except Exception as e:
        print(f"Error downloading repository: {e}")
        sys.exit(1)

    # Load metadata and config
    meta_path = model_dir / "meta.json"
    types_path = model_dir / "intervention_types.json"
    adapters_path = model_dir / "lora_adapters.pt"

    if not meta_path.exists() or not types_path.exists() or not adapters_path.exists():
        print(f"Error: Missing required model files (meta.json, intervention_types.json, or lora_adapters.pt) in {hf_repo}")
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(types_path, "r", encoding="utf-8") as f:
        intervention_types = json.load(f)

    config_dict = meta.get("config", {})
    original_base_model = config_dict.get("base_model", "dphn/dolphin-2.9-llama3-8b")

    # If llama sub-directory exists, use it as local base model
    local_llama = model_dir / "llama"
    if local_llama.exists() and (local_llama / "config.json").exists():
        print(f"Using local fine-tuned base model from: {local_llama}")
        config_dict["base_model"] = str(local_llama)
    else:
        print(f"Using pre-trained base model: {original_base_model}")
        config_dict["base_model"] = original_base_model

    config_dict["signal_dim"] = len(intervention_types)
    cfg = TrainingConfig(**config_dict)
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32

    # Load model
    print(f"Building SteeredModel on {model_device}...")
    model_obj = build_model(cfg, device=model_device)
    model_obj.lora.load_state_dict(
        torch.load(adapters_path, map_location="cpu")
    )
    model_obj.eval()

    default_steering_vector = torch.zeros(
        1, cfg.signal_dim, device=model_device, dtype=dtype
    )

    # Load tokenizer (prefer local files downloaded from repo if available, else load from original base model)
    print(f"Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load tokenizer from local cache ({e}). Loading from {original_base_model}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                original_base_model, trust_remote_code=True, token=hf_token if hf_token else None
            )
        except Exception as e_base:
            print(f"Error loading tokenizer: {e_base}")
            sys.exit(1)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build EOS token list
    eos_ids = [tokenizer.eos_token_id]
    for token in ["<|eot_id|>", "<|im_end|>"]:
        tok_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tok_id, int) and tok_id >= 0:
            eos_ids.append(tok_id)

    print(f"Runner successfully started and serving model: {hf_repo}")

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="", description="Optional model repo ID. Bypassed since this server runs a single model.")
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    stream: bool = Field(default=False)
    steering: Optional[Dict[str, float]] = Field(
        default=None,
        description="Per-request steering overrides, e.g. {'unpoliteness': 0.8}."
    )

class SteeringUpdateRequest(BaseModel):
    weights: Dict[str, float]

def find_dim_index(query: str, dims: List[str]) -> int:
    try:
        idx = int(query)
        if 0 <= idx < len(dims):
            return idx
    except ValueError:
        pass
    query_lower = query.lower().strip()
    matches = []
    for idx, name in enumerate(dims):
        if query_lower == name.lower():
            return idx
        if query_lower in name.lower():
            matches.append(idx)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous: '{query}' matches {[dims[i] for i in matches]}")
    raise ValueError(f"Dimension '{query}' not found.")

@app.get("/models")
@app.get("/v1/models")
async def list_models():
    if model_obj is None:
        raise HTTPException(status_code=503, detail="Server not ready yet.")
    return {
        "object": "list",
        "data": [
            {
                "id": hf_repo,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "steerable-model-runner",
            }
        ],
    }

@app.get("/models/info")
async def get_model_info():
    if model_obj is None:
        raise HTTPException(status_code=503, detail="Server not ready yet.")
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
    if model_obj is None:
        raise HTTPException(status_code=503, detail="Server not ready yet.")
    weights = {
        dim: default_steering_vector[0, idx].item()
        for idx, dim in enumerate(intervention_types)
    }
    return {"model": hf_repo, "dimensions": weights}

@app.post("/v1/steering")
async def update_steering(req: SteeringUpdateRequest):
    if model_obj is None:
        raise HTTPException(status_code=503, detail="Server not ready yet.")
    for dim_name, val in req.weights.items():
        try:
            idx = find_dim_index(dim_name, intervention_types)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not (0.0 <= val <= 1.0):
            raise HTTPException(status_code=400, detail=f"Weight for '{dim_name}' must be in [0.0, 1.0].")
        default_steering_vector[0, idx] = val
    weights = {dim: default_steering_vector[0, idx].item() for idx, dim in enumerate(intervention_types)}
    return {"status": "success", "model": hf_repo, "default_steering": weights}

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    if model_obj is None:
        raise HTTPException(status_code=503, detail="Server not ready yet.")

    # Build steering vector for this request
    req_s = default_steering_vector.clone()
    if request.steering:
        for dim_name, val in request.steering.items():
            try:
                idx = find_dim_index(dim_name, intervention_types)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            if not (0.0 <= val <= 1.0):
                raise HTTPException(status_code=400, detail=f"Steering '{dim_name}' must be in [0.0, 1.0].")
            req_s[0, idx] = val

    # Format messages using the tokenizer's chat template
    messages_list = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    input_ids = tokenizer.apply_chat_template(
        messages_list,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model_device)
    attention_mask = torch.ones_like(input_ids).to(model_device)

    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    steering_active = req_s.abs().sum().item() > 0.0

    def _gen_kwargs(streamer=None):
        kw = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=request.max_tokens,
            do_sample=True,
            temperature=request.temperature,
            top_p=request.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
            repetition_penalty=1.3,
            no_repeat_ngram_size=5,
            use_cache=False,
        )
        if streamer is not None:
            kw["streamer"] = streamer
        if request.top_k is not None and request.top_k > 0:
            kw["top_k"] = request.top_k
        return kw

    if request.stream:
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, clean_up_tokenization_spaces=True, skip_special_tokens=True
        )

        def run_generation():
            with generation_lock:
                if steering_active:
                    with active_steering(model_obj, req_s):
                        model_obj.llama.generate(**_gen_kwargs(streamer))
                else:
                    model_obj.llama.generate(**_gen_kwargs(streamer))

        thread = threading.Thread(target=run_generation)
        thread.start()

        async def event_generator():
            loop = asyncio.get_running_loop()
            try:
                while True:
                    token = await loop.run_in_executor(None, lambda: next(streamer, None))
                    if token is None:
                        break
                    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': hf_repo, 'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}]})}\n\n"
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': hf_repo, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': f'[Error: {e}]'}, 'finish_reason': 'error'}]})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                await loop.run_in_executor(None, thread.join)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    def run_non_stream():
        with generation_lock:
            if steering_active:
                with active_steering(model_obj, req_s):
                    return model_obj.llama.generate(**_gen_kwargs())
            else:
                return model_obj.llama.generate(**_gen_kwargs())

    loop = asyncio.get_running_loop()
    generated_ids = await loop.run_in_executor(None, run_non_stream)

    input_len = input_ids.shape[1]
    response_ids = generated_ids[0][input_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": hf_repo,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_len,
            "completion_tokens": len(response_ids),
            "total_tokens": input_len + len(response_ids),
        },
    }

if __name__ == "__main__":
    import uvicorn
    host_ip = os.environ.get("HOST", "0.0.0.0")
    port_num = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host_ip, port=port_num)
