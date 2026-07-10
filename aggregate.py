import argparse
import json
import os
import sys
import httpx
from typing import Dict, List, Optional
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(
    title="Steerable Model Runner Aggregator",
    description="Aggregates and proxies multiple steerable-model-runner instances.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Configuration ─────────────────────────────────────────────────────────────
backend_ports: List[int] = []
client = httpx.AsyncClient(timeout=600.0)

# Cached mapping of model ID -> backend URL (e.g. "http://localhost:8001")
model_to_backend_cache: Dict[str, str] = {}


# ── Helper functions ──────────────────────────────────────────────────────────

async def get_model_backends() -> Dict[str, str]:
    """Queries each port for their /v1/models to build a model ID -> URL mapping."""
    mapping = {}
    for port in backend_ports:
        backend_url = f"http://localhost:{port}"
        try:
            r = await client.get(f"{backend_url}/v1/models")
            if r.status_code == 200:
                data = r.json()
                for model in data.get("data", []):
                    model_id = model.get("id")
                    if model_id:
                        mapping[model_id] = backend_url
        except Exception as e:
            print(f"[aggregator] Warning: Failed to query backend on port {port}: {e}")
    return mapping


async def resolve_backend(model_name: str) -> Optional[str]:
    """Resolves the backend URL for a given model, refreshing cache if needed."""
    global model_to_backend_cache
    if model_name in model_to_backend_cache:
        return model_to_backend_cache[model_name]
    
    # Refresh cache
    model_to_backend_cache = await get_model_backends()
    return model_to_backend_cache.get(model_name)


async def proxy_request(backend_url: str, request: Request, cleaned_body: Optional[bytes] = None) -> Response:
    """Generic proxy helper that handles streaming and non-streaming requests."""
    path = request.url.path
    method = request.method
    headers = dict(request.headers)
    headers.pop("host", None)
    params = dict(request.query_params)
    
    if cleaned_body is not None:
        body = cleaned_body
    else:
        body = await request.body()
    
    url = f"{backend_url}{path}"
    
    # Check if this is a streaming completions call
    is_stream = False
    if method == "POST" and "chat/completions" in path:
        try:
            data = json.loads(body)
            is_stream = data.get("stream", False)
        except Exception:
            pass

    if is_stream:
        async def stream_generator():
            async with client.stream(method, url, headers=headers, params=params, content=body) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        try:
            r = await client.request(method, url, headers=headers, params=params, content=body)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Bad Gateway: Error connecting to backend: {exc}")
            
        res_headers = {}
        for k, v in r.headers.items():
            if k.lower() not in ("content-length", "content-encoding", "transfer-encoding", "connection"):
                res_headers[k] = v
        return Response(content=r.content, status_code=r.status_code, headers=res_headers)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/models")
@app.get("/v1/models")
async def list_models():
    """Aggregates list of models from all backends."""
    all_models = []
    seen_ids = set()
    
    for port in backend_ports:
        backend_url = f"http://localhost:{port}"
        try:
            r = await client.get(f"{backend_url}/v1/models")
            if r.status_code == 200:
                data = r.json()
                for model in data.get("data", []):
                    model_id = model.get("id")
                    if model_id and model_id not in seen_ids:
                        seen_ids.add(model_id)
                        all_models.append(model)
        except Exception as e:
            print(f"[aggregator] Failed listing models from port {port}: {e}")
            
    return {
        "object": "list",
        "data": all_models,
    }


@app.get("/models/info")
@app.get("/v1/models/info")
async def get_models_info(model: Optional[str] = None):
    """Returns model info. If model query param is omitted, returns info for all models."""
    if model:
        backend_url = await resolve_backend(model)
        if not backend_url:
            raise HTTPException(status_code=404, detail=f"Model '{model}' not found.")
        try:
            r = await client.get(f"{backend_url}/models/info")
            return r.json()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Error connecting to backend: {e}")
            
    # Combine info from all backends
    combined_info = {}
    for port in backend_ports:
        backend_url = f"http://localhost:{port}"
        try:
            r = await client.get(f"{backend_url}/models/info")
            if r.status_code == 200:
                info = r.json()
                model_id = info.get("id")
                if model_id:
                    combined_info[model_id] = info
        except Exception as e:
            print(f"[aggregator] Failed to get model info from port {port}: {e}")
    return combined_info


@app.get("/v1/steering")
async def get_steering(model: str):
    """Inspects steering for a specific model."""
    backend_url = await resolve_backend(model)
    if not backend_url:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found.")
    
    # Forward get request to the target backend
    try:
        r = await client.get(f"{backend_url}/v1/steering", params={"model": model})
        return Response(content=r.content, status_code=r.status_code, headers={"content-type": "application/json"})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error connecting to backend: {e}")


@app.post("/v1/steering")
async def update_steering(request: Request, model: Optional[str] = None):
    """Updates steering weights for a specific model."""
    body_bytes = await request.body()
    cleaned_body = body_bytes
    
    # Attempt to extract model from the JSON body if not in query parameters
    if not model and body_bytes:
        try:
            body_json = json.loads(body_bytes)
            model = body_json.pop("model", None)
            cleaned_body = json.dumps(body_json).encode("utf-8")
        except Exception:
            pass
            
    if not model:
        raise HTTPException(status_code=400, detail="Missing 'model' identifier in query parameter or body.")
        
    backend_url = await resolve_backend(model)
    if not backend_url:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found.")
        
    return await proxy_request(backend_url, request, cleaned_body=cleaned_body)


@app.post("/v1/chat/completions")
async def chat_completion(request: Request):
    """Routes completion request to the backend serving the specified model."""
    body_bytes = await request.body()
    model = None
    if body_bytes:
        try:
            body_json = json.loads(body_bytes)
            model = body_json.get("model")
        except Exception:
            pass
            
    if not model:
        raise HTTPException(status_code=400, detail="Missing 'model' field in chat completion request.")
        
    backend_url = await resolve_backend(model)
    if not backend_url:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found in any backend.")
        
    return await proxy_request(backend_url, request)


# ── Startup/CLI Entrypoint ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Run the Steerable Model Runner Aggregator Proxy.")
    parser.add_argument(
        "--ports",
        nargs="+",
        type=int,
        required=True,
        help="List of ports of the backend steerable-model-runner instances."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the aggregator server to."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Port to run the aggregator server on."
    )
    
    args, unknown = parser.parse_known_args()
    
    # Handle list of ports that might have been passed as comma-separated values (e.g. from env or CLI)
    expanded_ports = []
    for p in args.ports:
        expanded_ports.append(p)
        
    backend_ports = expanded_ports
    print(f"[aggregator] Initializing aggregator on {args.host}:{args.port}")
    print(f"[aggregator] Aggregating backend ports: {backend_ports}")
    
    uvicorn.run(app, host=args.host, port=args.port)
