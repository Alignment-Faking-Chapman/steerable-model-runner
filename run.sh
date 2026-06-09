#!/bin/bash
# run.sh - Build and run the Steerable Model Runner in Docker.

set -e

# Resolve script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME="steerable-model-runner"
HF_REPO_VAL="${HF_REPO}"
HF_TOKEN_VAL="${HF_TOKEN}"
PORT_VAL="${PORT:-8000}"
COMPILED_ADAPTER_VAL="${COMPILED_ADAPTER}"

# Parse options to extract parameters if provided
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-repo=*|--hf_repo=*)
      HF_REPO_VAL="${1#*=}"
      shift
      ;;
    --hf-repo|--hf_repo)
      if [[ -n "$2" && "$2" != -* ]]; then
        HF_REPO_VAL="$2"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --hf-token=*|--hf_token=*)
      HF_TOKEN_VAL="${1#*=}"
      shift
      ;;
    --hf-token|--hf_token)
      if [[ -n "$2" && "$2" != -* ]]; then
        HF_TOKEN_VAL="$2"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --compiled-adapter=*|--compiled_adapter=*)
      COMPILED_ADAPTER_VAL="${1#*=}"
      shift
      ;;
    --compiled-adapter|--compiled_adapter)
      if [[ -n "$2" && "$2" != -* ]]; then
        COMPILED_ADAPTER_VAL="$2"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --port=*|--port=*)
      PORT_VAL="${1#*=}"
      shift
      ;;
    --port)
      if [[ -n "$2" && "$2" != -* ]]; then
        PORT_VAL="$2"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    *)
      echo "Error: Unknown argument $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${HF_REPO_VAL}" ]]; then
  echo "Error: HF_REPO is required to run the server."
  echo "Usage: ./run.sh --hf-repo \"ChapAF/steerable-dolphin-8b\" [--hf-token \"your_token\"] [--port 8000] [--compiled-adapter \"path/to/compiled_adapter.pt\"]"
  exit 1
fi

echo -e "\033[1;34mBuilding Docker image '${IMAGE_NAME}'...\033[0m"
docker build -t "${IMAGE_NAME}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"

HOST_HF_CACHE="${HOME}/.cache/huggingface"
mkdir -p "${HOST_HF_CACHE}"

EXTRA_RUN_ARGS=()
if [[ -n "${COMPILED_ADAPTER_VAL}" ]]; then
  # Resolve to absolute path
  ABS_ADAPTER_PATH="$(cd "$(dirname "${COMPILED_ADAPTER_VAL}")" && pwd)/$(basename "${COMPILED_ADAPTER_VAL}")"
  if [[ ! -f "${ABS_ADAPTER_PATH}" ]]; then
    echo "Error: Compiled adapter file not found: ${ABS_ADAPTER_PATH}" >&2
    exit 1
  fi
  echo -e "\033[1;36mUsing compiled prompt adapter: ${ABS_ADAPTER_PATH}\033[0m"
  EXTRA_RUN_ARGS+=("-v" "${ABS_ADAPTER_PATH}:/app/compiled_adapter.pt" "-e" "COMPILED_ADAPTER=/app/compiled_adapter.pt")
fi

echo -e "\033[1;32mStarting Steerable Model Runner for HF repo: ${HF_REPO_VAL} on port ${PORT_VAL}...\033[0m"
docker run --gpus all --rm \
  -v "${HOST_HF_CACHE}:/root/.cache/huggingface" \
  -e HF_REPO="${HF_REPO_VAL}" \
  -e HF_TOKEN="${HF_TOKEN_VAL}" \
  -e PORT="${PORT_VAL}" \
  "${EXTRA_RUN_ARGS[@]}" \
  -p "${PORT_VAL}:${PORT_VAL}" \
  "${IMAGE_NAME}"
