# Steerable Model Runner 🚀

A FastAPI-based server designed to run and serve Large Language Models (LLMs) with dynamic, runtime-steerable LoRA adapters. This server is compatible with OpenAI's Chat Completions API and adds custom endpoints for inspecting and adjusting steering vectors on the fly.

---

## Key Features
- **OpenAI-Compatible Chat API**: Drop-in replacement for OpenAI endpoints (`/v1/chat/completions`) supporting both streaming and standard responses.
- **Dynamic Run-time Steering**: Steer model behaviors (e.g., politeness, helpfulness, style) per-request or globally by adjusting weights.
- **Docker Ready**: Easy setup and deployment via a unified Docker launch script with GPU support.
- **Auto-downloading**: Seamlessly fetches models directly from Hugging Face Hub.

---

## Quick Start

### Prerequisites
- Docker with NVIDIA Container Toolkit (for GPU acceleration) OR a PyTorch environment with CUDA.

### Running with Docker (Recommended)
Use the `./run.sh` helper script to automatically build the Docker image and run the container with GPU support. 

> [!TIP]
> **Model Caching**: The script automatically mounts the host's Hugging Face cache directory (`~/.cache/huggingface`) to the container. Subsequent runs with the same model repository or base model will load almost instantly from your local cache without re-downloading files.

```bash
# Run with a Hugging Face repository
./run.sh --hf-repo "ChapAF/steerable-dolphin-8b" --hf-token "your_hf_token" --port 8000
```

#### Command-Line Arguments
The `run.sh` script supports the following command-line flags (which fallback to environment variables if not provided):

| Flag | Env Var | Description | Required | Default |
|---|---|---|---|---|
| `--hf-repo` / `--hf_repo` | `HF_REPO` | The Hugging Face Repository ID | **Yes** | - |
| `--hf-token` / `--hf_token` | `HF_TOKEN` | Hugging Face Access Token (for gated repositories) | No | - |
| `--port` | `PORT` | Local host port to bind to the server | No | `8000` |

---

## Repository Requirements

The target Hugging Face repository must contain the following steerable model files:
- **`meta.json`**: Holds training configuration metadata.
- **`intervention_types.json`**: List of steering dimensions (e.g. `["unpoliteness", "sycophancy"]`).
- **`lora_adapters.pt`**: The trained steerable LoRA weights.
- **`llama/` (Optional)**: A subdirectory containing the base model config and weights. If missing, the runner will load the pre-trained base model specified in `meta.json`.

---

## API Endpoints

### 1. Model Info
* **List Models (`GET /v1/models` or `GET /models`)**
  Returns details of the currently loaded model.
* **Get Model Details (`GET /models/info`)**
  Returns the loaded repository name, number of steering dimensions, list of dimensions, and their current default weights.

---

### 2. Steering Management
You can read and update the default global steering vector used for all requests.

* **Get Global Steering (`GET /v1/steering`)**
  ```json
  {
    "model": "ChapAF/steerable-dolphin-8b",
    "dimensions": {
      "unpoliteness": 0.0,
      "sycophancy": 0.0
    }
  }
  ```

* **Update Global Steering (`POST /v1/steering`)**
  Change the default steering weights globally. Weights must be in the range `[0.0, 1.0]`.
  ```json
  {
    "weights": {
      "unpoliteness": 0.5
    }
  }
  ```

---

### 3. Chat Completions (`POST /v1/chat/completions`)
Generate text using an OpenAI-compatible request body. You can also specify per-request steering parameter overrides in the payload.

#### Example Request
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Why is the sky blue?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 256,
  "stream": false,
  "steering": {
    "unpoliteness": 0.8
  }
}
```

#### Example cURL Command
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a story."}],
    "steering": {"unpoliteness": 0.2}
  }'
```

---

## System Architecture

- **`server.py`**: A FastAPI application that handles request routing, Hugging Face repository downloading, tokenization, streaming responses, and locking for concurrent generation requests.
- **`model.py`**: Defines the custom `AdaptedLinear` PyTorch modules and `SteeredModel` wrapper. Dynamically swaps standard linear layers of Llama models for LoRA adapters that scale output perturbations based on the input steering signal.
- **`config.py`**: Training configuration class and mapping utilities.
