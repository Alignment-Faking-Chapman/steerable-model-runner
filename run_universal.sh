#!/bin/bash
#SBATCH --job-name=steerable-runner
#SBATCH --output=logs_%j.out
#SBATCH --error=logs_%j.err
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# run.sh - Build and run the Steerable Model Runner.
# Prefers Docker if it's available; falls back to submitting a SLURM job
# (via sbatch, re-invoking this same script) if Docker is not available.
#
# The #SBATCH header above is only honored when this script is *submitted*
# with `sbatch run.sh ...`; it is ignored (treated as comments) when the
# script is run directly with `./run.sh ...` or `bash run.sh ...`.

set -e

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
IMAGE_NAME="steerable-model-runner"
CONDA_ENV_NAME="steerable_runner_env"

HF_REPO_VAL="${HF_REPO}"
HF_TOKEN_VAL="${HF_TOKEN}"
PORT_VAL="${PORT:-8000}"
COMPILED_ADAPTER_VAL="${COMPILED_ADAPTER}"
GPUS_VAL="${GPUS:-1}"

# ---------------------------------------------------------------------------
# 1. Parse command-line arguments (no-ops on the SLURM re-invocation, since
#    sbatch passes values down via --export instead of argv)
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-repo=*|--hf_repo=*)
      HF_REPO_VAL="${1#*=}"; shift ;;
    --hf-repo|--hf_repo)
      if [[ -n "$2" && "$2" != -* ]]; then HF_REPO_VAL="$2"; shift 2
      else echo "Error: Argument for $1 is missing" >&2; exit 1; fi ;;
    --hf-token=*|--hf_token=*)
      HF_TOKEN_VAL="${1#*=}"; shift ;;
    --hf-token|--hf_token)
      if [[ -n "$2" && "$2" != -* ]]; then HF_TOKEN_VAL="$2"; shift 2
      else echo "Error: Argument for $1 is missing" >&2; exit 1; fi ;;
    --compiled-adapter=*|--compiled_adapter=*)
      COMPILED_ADAPTER_VAL="${1#*=}"; shift ;;
    --compiled-adapter|--compiled_adapter)
      if [[ -n "$2" && "$2" != -* ]]; then COMPILED_ADAPTER_VAL="$2"; shift 2
      else echo "Error: Argument for $1 is missing" >&2; exit 1; fi ;;
    --port=*)
      PORT_VAL="${1#*=}"; shift ;;
    --port)
      if [[ -n "$2" && "$2" != -* ]]; then PORT_VAL="$2"; shift 2
      else echo "Error: Argument for $1 is missing" >&2; exit 1; fi ;;
    --gpus=*)
      GPUS_VAL="${1#*=}"; shift ;;
    --gpus)
      if [[ -n "$2" && "$2" != -* ]]; then GPUS_VAL="$2"; shift 2
      else echo "Error: Argument for $1 is missing" >&2; exit 1; fi ;;
    *)
      echo "Error: Unknown argument $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# 2. If we're not already inside a SLURM allocation, this is the "launcher"
#    invocation: validate input, then choose Docker or SLURM.
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID}" ]]; then

  if [[ -z "${HF_REPO_VAL}" ]]; then
    echo "Error: HF_REPO is required to run the server."
    echo "Usage: ./run.sh --hf-repo \"ChapAF/steerable-dolphin-8b\" [--hf-token \"your_token\"] [--port 8000] [--compiled-adapter \"path/to/compiled_adapter.pt\"] [--gpus \"1\"]"
    exit 1
  fi

  if [[ -n "${COMPILED_ADAPTER_VAL}" ]]; then
    ABS_ADAPTER_PATH="$(cd "$(dirname "${COMPILED_ADAPTER_VAL}")" && pwd)/$(basename "${COMPILED_ADAPTER_VAL}")"
    if [[ ! -f "${ABS_ADAPTER_PATH}" ]]; then
      echo "Error: Compiled adapter file not found: ${ABS_ADAPTER_PATH}" >&2
      exit 1
    fi
    COMPILED_ADAPTER_VAL="${ABS_ADAPTER_PATH}"
  fi

  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    # ------------------------------------------------------------------- #
    # Docker path (preferred)
    # ------------------------------------------------------------------- #
    echo -e "\033[1;34mDocker detected. Building image '${IMAGE_NAME}'...\033[0m"
    docker build -t "${IMAGE_NAME}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"

    HOST_HF_CACHE="${HOME}/.cache/huggingface"
    mkdir -p "${HOST_HF_CACHE}"

    EXTRA_RUN_ARGS=()
    if [[ -n "${COMPILED_ADAPTER_VAL}" ]]; then
      echo -e "\033[1;36mUsing compiled prompt adapter: ${COMPILED_ADAPTER_VAL}\033[0m"
      EXTRA_RUN_ARGS+=("-v" "${COMPILED_ADAPTER_VAL}:/app/compiled_adapter.pt" "-e" "COMPILED_ADAPTER=/app/compiled_adapter.pt")
    fi

    echo -e "\033[1;32mStarting Steerable Model Runner for HF repo: ${HF_REPO_VAL} on port ${PORT_VAL}...\033[0m"
    docker run --gpus "${GPUS_VAL}" --rm \
      -v "${HOST_HF_CACHE}:/root/.cache/huggingface" \
      -e HF_REPO="${HF_REPO_VAL}" \
      -e HF_TOKEN="${HF_TOKEN_VAL}" \
      -e PORT="${PORT_VAL}" \
      "${EXTRA_RUN_ARGS[@]}" \
      -p "${PORT_VAL}:${PORT_VAL}" \
      "${IMAGE_NAME}"
    exit 0

  elif command -v sbatch >/dev/null 2>&1; then
    # ------------------------------------------------------------------- #
    # SLURM path (fallback, no Docker available)
    # ------------------------------------------------------------------- #
    echo -e "\033[1;32mDocker not available. Submitting job to SLURM gpuq partition...\033[0m"
    echo "Repo: ${HF_REPO_VAL}"
    echo "Port: ${PORT_VAL}"
    sbatch --export=ALL,HF_REPO="${HF_REPO_VAL}",HF_TOKEN="${HF_TOKEN_VAL}",PORT="${PORT_VAL}",COMPILED_ADAPTER="${COMPILED_ADAPTER_VAL}" "${SCRIPT_PATH}"
    exit 0

  else
    echo "Error: Neither Docker nor SLURM (sbatch) is available on this machine." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# 3. We're inside the SLURM allocation (SLURM_JOB_ID is set): set up the
#    conda environment and start the server.
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR}"

module load cuda11.8

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    echo "Creating conda environment '${CONDA_ENV_NAME}'..."
    conda create -y -n "${CONDA_ENV_NAME}" python=3.12.4
    conda activate "${CONDA_ENV_NAME}"
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
    pip install --no-cache-dir -r requirements.txt
else
    echo "Conda environment '${CONDA_ENV_NAME}' already exists. Activating..."
    conda activate "${CONDA_ENV_NAME}"
fi

echo "Starting Steerable Model Runner for ${HF_REPO} on port ${PORT}..."
exec python3 server.py
