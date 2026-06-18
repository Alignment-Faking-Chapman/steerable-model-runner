# vLLM ships its own CUDA environment; use the official vLLM image.
FROM vllm/vllm-openai:latest

WORKDIR /app

# Install remaining Python dependencies (fastapi, uvicorn, huggingface_hub, etc.)
# vLLM and transformers are already bundled in the base image.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy runner source code
COPY . .

# Install the package in editable mode so the vllm.general_plugins entry point
# (declared in pyproject.toml) is registered in every vLLM worker process.
RUN pip install --no-cache-dir -e .

# Expose FastAPI port
EXPOSE 8000

ENTRYPOINT ["python3", "server.py"]
