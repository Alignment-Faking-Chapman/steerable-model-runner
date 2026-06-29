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
GPUS_VAL="${GPUS:-2}"
MAX_GPUS_VAL="${MAX_GPUS:-2}"
MIN_GPU_MEMORY_VAL="${MIN_GPU_MEMORY:-4000}"
BG_VAL="false"
NO_RM_VAL="false"
QUANTIZATION_VAL="${QUANTIZATION}"
GPU_MEMORY_UTILIZATION_VAL="${GPU_MEMORY_UTILIZATION}"
MAX_MODEL_LEN_VAL="${MAX_MODEL_LEN}"
TENSOR_PARALLEL_SIZE_VAL="${TENSOR_PARALLEL_SIZE}"
PIPELINE_PARALLEL_SIZE_VAL="${PIPELINE_PARALLEL_SIZE}"

# ---------------------------------------------------------------------------
# 1. Parse command-line arguments (no-ops on the SLURM re-invocation, since
#    sbatch passes values down via --export instead of argv)
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bg)
      BG_VAL="true"; shift ;;
    --no-rm|--no_rm)
      NO_RM_VAL="true"; shift ;;
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
    --max-gpus=*|--max_gpus=*)
      MAX_GPUS_VAL="${1#*=}"; shift ;;
    --max-gpus|--max_gpus)
      if [[ -n "$2" && "$2" != -* ]]; then MAX_GPUS_VAL="$2"; shift 2
      else echo "Error: Argument for $1 is missing" >&2; exit 1; fi ;;
    --min-gpu-memory=*|--min_gpu_memory=*)
      MIN_GPU_MEMORY_VAL="${1#*=}"; shift ;;
    --min-gpu-memory|--min_gpu_memory)
      if [[ -n "$2" && "$2" != -* ]]; then MIN_GPU_MEMORY_VAL="$2"; shift 2
      else echo "Error: Argument for $1 is missing" >&2; exit 1; fi ;;
    --quantization=*|--quantization=*)
      QUANTIZATION_VAL="${1#*=}"; shift ;;
    --quantization)
      if [[ -n "$2" && "$2" != -* ]]; then QUANTIZATION_VAL="$2"; shift 2
      else echo "Error: Argument for $1 is missing" >&2; exit 1; fi ;;
    --gpu-memory-utilization=*|--gpu_memory_utilization=*)
      GPU_MEMORY_UTILIZATION_VAL="${1#*=}"; shift ;;
    --gpu-memory-utilization|--gpu_memory_utilization)
      if [[ -n "$2" && "$2" != -* ]]; then GPU_MEMORY_UTILIZATION_VAL="$2"; shift 2
      else echo "Error: Argument for $1 is missing" >&2; exit 1; fi ;;
    --max-model-len=*|--max_model_len=*)
      MAX_MODEL_LEN_VAL="${1#*=}"; shift ;;
    --max-model-len|--max_model_len)
      if [[ -n "$2" && "$2" != -* ]]; then MAX_MODEL_LEN_VAL="$2"; shift 2
      else echo "Error: Argument for $1 is missing" >&2; exit 1; fi ;;
    --tensor-parallel-size=*|--tensor_parallel_size=*)
      TENSOR_PARALLEL_SIZE_VAL="${1#*=}"; shift ;;
    --tensor-parallel-size|--tensor_parallel_size)
      if [[ -n "$2" && "$2" != -* ]]; then TENSOR_PARALLEL_SIZE_VAL="$2"; shift 2
      else echo "Error: Argument for $1 is missing" >&2; exit 1; fi ;;
    --pipeline-parallel-size=*|--pipeline_parallel_size=*)
      PIPELINE_PARALLEL_SIZE_VAL="${1#*=}"; shift ;;
    --pipeline-parallel-size|--pipeline_parallel_size)
      if [[ -n "$2" && "$2" != -* ]]; then PIPELINE_PARALLEL_SIZE_VAL="$2"; shift 2
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
    echo "Usage: ./run.sh --hf-repo \"ChapAF/steerable-dolphin-8b\" [--hf-token \"your_token\"] [--port 8000] [--compiled-adapter \"path/to/compiled_adapter.pt\"] [--gpus \"2\"]"
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

    # Determine which GPUs to use
    if command -v nvidia-smi >/dev/null 2>&1; then
      if [[ "${GPUS_VAL}" == device=* ]]; then
        echo -e "\033[1;36mUsing user-specified GPU devices: ${GPUS_VAL}\033[0m"
      else
        gpu_list=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | sort -t ',' -k 2 -n -r || true)
        
        if [[ -z "${gpu_list}" ]]; then
          echo -e "\033[1;33mWarning: Failed to query GPU list via nvidia-smi. Falling back to GPUS_VAL: ${GPUS_VAL}\033[0m"
        else
          usable_gpus=()
          min_mem="${MIN_GPU_MEMORY_VAL}"
          
          while IFS=',' read -r idx free_mem; do
            idx=$(echo "${idx}" | xargs)
            free_mem=$(echo "${free_mem}" | xargs)
            
            if [[ -n "${idx}" && -n "${free_mem}" ]]; then
              if (( free_mem >= min_mem )); then
                usable_gpus+=("${idx}")
              fi
            fi
          done <<< "${gpu_list}"
          
          num_usable=${#usable_gpus[@]}
          if [[ "${num_usable}" -eq 0 ]]; then
            echo -e "\033[1;33mWarning: No GPUs found with at least ${min_mem} MiB of free memory. Falling back to GPUS_VAL: ${GPUS_VAL}\033[0m"
          else
            target_count=""
            if [[ "${GPUS_VAL}" =~ ^[0-9]+$ ]]; then
              target_count="${GPUS_VAL}"
            elif [[ "${GPUS_VAL}" == "all" ]]; then
              target_count="${num_usable}"
            else
              target_count=1
            fi
            
            if [[ -n "${MAX_GPUS_VAL}" && "${MAX_GPUS_VAL}" =~ ^[0-9]+$ ]]; then
              if (( target_count > MAX_GPUS_VAL )); then
                echo -e "\033[1;36mCapping requested GPU count (${target_count}) to max GPU count (${MAX_GPUS_VAL})\033[0m"
                target_count="${MAX_GPUS_VAL}"
              fi
            fi
            
            if (( target_count > num_usable )); then
              echo -e "\033[1;33mWarning: Requested ${target_count} GPUs, but only ${num_usable} GPUs have at least ${min_mem} MiB free memory.\033[0m"
              echo -e "\033[1;33mUsing the ${num_usable} available GPUs.\033[0m"
              target_count="${num_usable}"
            fi
            
            selected_gpus=("${usable_gpus[@]:0:target_count}")
            selected_str=$(IFS=,; echo "${selected_gpus[*]}")
            
            echo -e "\033[1;36mAutomatically selected GPUs: ${selected_str} (sorted by free memory descending)\033[0m"
            GPUS_VAL="device=${selected_str}"
          fi
        fi
      fi
    else
      echo -e "\033[1;33mnvidia-smi not available on host. Passing GPUS_VAL directly: ${GPUS_VAL}\033[0m"
    fi

    # If the GPU specification contains commas (e.g. multiple devices), Docker requires
    # it to be enclosed in literal double quotes to avoid parsing errors.
    if [[ "${GPUS_VAL}" == *,* ]]; then
      if [[ "${GPUS_VAL}" != \"*\" && "${GPUS_VAL}" != \'*\' ]]; then
        GPUS_VAL="\"${GPUS_VAL}\""
      fi
    fi

    echo -e "\033[1;32mStarting Steerable Model Runner for HF repo: ${HF_REPO_VAL} on port ${PORT_VAL}...\033[0m"
    DOCKER_RUN_ARGS=()
    if [[ "${BG_VAL}" == "true" ]]; then
      DOCKER_RUN_ARGS+=("-d")
    fi
    if [[ "${NO_RM_VAL}" != "true" ]]; then
      DOCKER_RUN_ARGS+=("--rm")
    fi

    docker run "${DOCKER_RUN_ARGS[@]}" --gpus "${GPUS_VAL}" \
      -v "${HOST_HF_CACHE}:/root/.cache/huggingface" \
      -e HF_REPO="${HF_REPO_VAL}" \
      -e HF_TOKEN="${HF_TOKEN_VAL}" \
      -e PORT="${PORT_VAL}" \
      -e QUANTIZATION="${QUANTIZATION_VAL}" \
      -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_VAL}" \
      -e MAX_MODEL_LEN="${MAX_MODEL_LEN_VAL}" \
      -e TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE_VAL}" \
      -e PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE_VAL}" \
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
    sbatch --gres=gpu:${GPUS_VAL} --export=ALL,HF_REPO="${HF_REPO_VAL}",HF_TOKEN="${HF_TOKEN_VAL}",PORT="${PORT_VAL}",COMPILED_ADAPTER="${COMPILED_ADAPTER_VAL}",QUANTIZATION="${QUANTIZATION_VAL}",GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_VAL}",MAX_MODEL_LEN="${MAX_MODEL_LEN_VAL}",TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE_VAL}",PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE_VAL}" "${SCRIPT_PATH}"
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
