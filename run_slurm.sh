#!/bin/bash
set -e

# Resolve script directory and name
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"

# 1. Parse incoming command-line arguments
HF_REPO_VAL="${HF_REPO}"
HF_TOKEN_VAL="${HF_TOKEN}"
PORT_VAL="${PORT:-8000}"
COMPILED_ADAPTER_VAL="${COMPILED_ADAPTER}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-repo=*|--hf_repo=*) HF_REPO_VAL="${1#*=}"; shift ;;
    --hf-repo|--hf_repo) HF_REPO_VAL="$2"; shift 2 ;;
    --hf-token=*|--hf_token=*) HF_TOKEN_VAL="${1#*=}"; shift ;;
    --hf-token|--hf_token) HF_TOKEN_VAL="$2"; shift 2 ;;
    --compiled-adapter=*|--compiled_adapter=*) COMPILED_ADAPTER_VAL="${1#*=}"; shift ;;
    --compiled-adapter|--compiled_adapter) COMPILED_ADAPTER_VAL="$2"; shift 2 ;;
    --port=*|--port=*) PORT_VAL="${1#*=}"; shift ;;
    --port) PORT_VAL="$2"; shift 2 ;;
    *) echo "Error: Unknown argument $1" >&2; exit 1 ;;
  esac
done

# 2. Check if we are running locally on the login node or inside the SLURM allocation
if [[ -z "${SLURM_JOB_ID}" ]]; then
  # =========================================================================
  # RUNNING ON LOGIN NODE: Validate input and submit to cluster queue via sbatch
  # =========================================================================
  if [[ -z "${HF_REPO_VAL}" ]]; then
    echo "Error: HF_REPO is required."
    echo "Usage: ./run.sh --hf-repo \"ChapAF/steerable-dolphin-8b\" [--port 8000]"
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

  echo -e "\033[1;32mSubmitting job to SLURM gpuq partition...\033[0m"
  echo "Repo: ${HF_REPO_VAL}"
  echo "Port: ${PORT_VAL}"

  # Re-invoke this exact script using sbatch, passing variables down via --export
  sbatch --export=ALL,HF_REPO="${HF_REPO_VAL}",HF_TOKEN="${HF_TOKEN_VAL}",PORT="${PORT_VAL}",COMPILED_ADAPTER="${COMPILED_ADAPTER_VAL}" "${SCRIPT_PATH}"
  exit 0
fi

# =========================================================================
# RUNNING INSIDE COMPUTE NODE: Set up environment and start application
# =========================================================================
cd "${SLURM_SUBMIT_DIR}"

# Load local cluster dependencies
module purge
module load python/3.10

# Fast check virtual environment setup
VENV_DIR="${SLURM_SUBMIT_DIR}/.venv_slurm"

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating a fresh virtual environment at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip
    pip install --no-cache-dir -r requirements.txt
else
    echo "Virtual environment already exists. Activating..."
    source "${VENV_DIR}/bin/activate"
fi

echo "Starting Steerable Model Runner for ${HF_REPO} on port ${PORT}..."
exec python3 server.py
#SBATCH --job-name=steerable-runner
#SBATCH --output=logs_%j.out
#SBATCH --error=logs_%j.err
#SBATCH --partition=gpuq            # Target bright81's GPU queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --cpus-per-task=4           
#SBATCH --mem=32G                   
#SBATCH --time=1-00:00:00           # Max 1-day limit for gpuq

set -e

# Resolve script directory and name
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"

# 1. Parse incoming command-line arguments
HF_REPO_VAL="${HF_REPO}"
HF_TOKEN_VAL="${HF_TOKEN}"
PORT_VAL="${PORT:-8000}"
COMPILED_ADAPTER_VAL="${COMPILED_ADAPTER}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-repo=*|--hf_repo=*) HF_REPO_VAL="${1#*=}"; shift ;;
    --hf-repo|--hf_repo) HF_REPO_VAL="$2"; shift 2 ;;
    --hf-token=*|--hf_token=*) HF_TOKEN_VAL="${1#*=}"; shift ;;
    --hf-token|--hf_token) HF_TOKEN_VAL="$2"; shift 2 ;;
    --compiled-adapter=*|--compiled_adapter=*) COMPILED_ADAPTER_VAL="${1#*=}"; shift ;;
    --compiled-adapter|--compiled_adapter) COMPILED_ADAPTER_VAL="$2"; shift 2 ;;
    --port=*|--port=*) PORT_VAL="${1#*=}"; shift ;;
    --port) PORT_VAL="$2"; shift 2 ;;
    *) echo "Error: Unknown argument $1" >&2; exit 1 ;;
  esac
done

# 2. Check if we are running locally on the login node or inside the SLURM allocation
if [[ -z "${SLURM_JOB_ID}" ]]; then
  # =========================================================================
  # RUNNING ON LOGIN NODE: Validate input and submit to cluster queue via sbatch
  # =========================================================================
  if [[ -z "${HF_REPO_VAL}" ]]; then
    echo "Error: HF_REPO is required."
    echo "Usage: ./run.sh --hf-repo \"ChapAF/steerable-dolphin-8b\" [--port 8000]"
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

  echo -e "\033[1;32mSubmitting job to SLURM gpuq partition...\033[0m"
  echo "Repo: ${HF_REPO_VAL}"
  echo "Port: ${PORT_VAL}"

  # Re-invoke this exact script using sbatch, passing variables down via --export
  sbatch --export=ALL,HF_REPO="${HF_REPO_VAL}",HF_TOKEN="${HF_TOKEN_VAL}",PORT="${PORT_VAL}",COMPILED_ADAPTER="${COMPILED_ADAPTER_VAL}" "${SCRIPT_PATH}"
  exit 0
fi

# =========================================================================
# RUNNING INSIDE COMPUTE NODE: Set up environment and start application
# =========================================================================
cd "${SLURM_SUBMIT_DIR}"

# Load local cluster dependencies
module purge
module load python/3.10
module load cuda/12.4

# Fast check virtual environment setup
VENV_DIR="${SLURM_SUBMIT_DIR}/.venv_slurm"

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating a fresh virtual environment at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip
    pip install --no-cache-dir -r requirements.txt
else
    echo "Virtual environment already exists. Activating..."
    source "${VENV_DIR}/bin/activate"
fi

echo "Starting Steerable Model Runner for ${HF_REPO} on port ${PORT}..."
exec python3 server.py

#SBATCH --job-name=steerable-runner
#SBATCH --output=logs_%j.out
#SBATCH --error=logs_%j.err
#SBATCH --partition=gpuq            # Target bright81's GPU queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --cpus-per-task=4           
#SBATCH --mem=32G                   
#SBATCH --time=1-00:00:00           # Max 1-day limit for gpuq

set -e

# Resolve script directory and name
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"

# 1. Parse incoming command-line arguments
HF_REPO_VAL="${HF_REPO}"
HF_TOKEN_VAL="${HF_TOKEN}"
PORT_VAL="${PORT:-8000}"
COMPILED_ADAPTER_VAL="${COMPILED_ADAPTER}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-repo=*|--hf_repo=*) HF_REPO_VAL="${1#*=}"; shift ;;
    --hf-repo|--hf_repo) HF_REPO_VAL="$2"; shift 2 ;;
    --hf-token=*|--hf_token=*) HF_TOKEN_VAL="${1#*=}"; shift ;;
    --hf-token|--hf_token) HF_TOKEN_VAL="$2"; shift 2 ;;
    --compiled-adapter=*|--compiled_adapter=*) COMPILED_ADAPTER_VAL="${1#*=}"; shift ;;
    --compiled-adapter|--compiled_adapter) COMPILED_ADAPTER_VAL="$2"; shift 2 ;;
    --port=*|--port=*) PORT_VAL="${1#*=}"; shift ;;
    --port) PORT_VAL="$2"; shift 2 ;;
    *) echo "Error: Unknown argument $1" >&2; exit 1 ;;
  esac
done

# 2. Check if we are running locally on the login node or inside the SLURM allocation
if [[ -z "${SLURM_JOB_ID}" ]]; then
  # =========================================================================
  # RUNNING ON LOGIN NODE: Validate input and submit to cluster queue via sbatch
  # =========================================================================
  if [[ -z "${HF_REPO_VAL}" ]]; then
    echo "Error: HF_REPO is required."
    echo "Usage: ./run.sh --hf-repo \"ChapAF/steerable-dolphin-8b\" [--port 8000]"
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

  echo -e "\033[1;32mSubmitting job to SLURM gpuq partition...\033[0m"
  echo "Repo: ${HF_REPO_VAL}"
  echo "Port: ${PORT_VAL}"

  # Re-invoke this exact script using sbatch, passing variables down via --export
  sbatch --export=ALL,HF_REPO="${HF_REPO_VAL}",HF_TOKEN="${HF_TOKEN_VAL}",PORT="${PORT_VAL}",COMPILED_ADAPTER="${COMPILED_ADAPTER_VAL}" "${SCRIPT_PATH}"
  exit 0
fi

# =========================================================================
# RUNNING INSIDE COMPUTE NODE: Set up environment and start application
# =========================================================================
cd "${SLURM_SUBMIT_DIR}"

# Load local cluster dependencies
module purge
module load python/3.10
module load cuda/12.4

# Fast check virtual environment setup
VENV_DIR="${SLURM_SUBMIT_DIR}/.venv_slurm"

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating a fresh virtual environment at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip
    pip install --no-cache-dir -r requirements.txt
else
    echo "Virtual environment already exists. Activating..."
    source "${VENV_DIR}/bin/activate"
fi

echo "Starting Steerable Model Runner for ${HF_REPO} on port ${PORT}..."
exec python3 server.py
#SBATCH --job-name=steerable-runner
#SBATCH --output=logs_%j.out
#SBATCH --error=logs_%j.err
#SBATCH --partition=gpuq            # Target bright81's GPU queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --cpus-per-task=4           
#SBATCH --mem=32G                   
#SBATCH --time=1-00:00:00           # Max 1-day limit for gpuq

set -e

# Resolve script directory and name
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"

# 1. Parse incoming command-line arguments
HF_REPO_VAL="${HF_REPO}"
HF_TOKEN_VAL="${HF_TOKEN}"
PORT_VAL="${PORT:-8000}"
COMPILED_ADAPTER_VAL="${COMPILED_ADAPTER}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-repo=*|--hf_repo=*) HF_REPO_VAL="${1#*=}"; shift ;;
    --hf-repo|--hf_repo) HF_REPO_VAL="$2"; shift 2 ;;
    --hf-token=*|--hf_token=*) HF_TOKEN_VAL="${1#*=}"; shift ;;
    --hf-token|--hf_token) HF_TOKEN_VAL="$2"; shift 2 ;;
    --compiled-adapter=*|--compiled_adapter=*) COMPILED_ADAPTER_VAL="${1#*=}"; shift ;;
    --compiled-adapter|--compiled_adapter) COMPILED_ADAPTER_VAL="$2"; shift 2 ;;
    --port=*|--port=*) PORT_VAL="${1#*=}"; shift ;;
    --port) PORT_VAL="$2"; shift 2 ;;
    *) echo "Error: Unknown argument $1" >&2; exit 1 ;;
  esac
done

# 2. Check if we are running locally on the login node or inside the SLURM allocation
if [[ -z "${SLURM_JOB_ID}" ]]; then
  # =========================================================================
  # RUNNING ON LOGIN NODE: Validate input and submit to cluster queue via sbatch
  # =========================================================================
  if [[ -z "${HF_REPO_VAL}" ]]; then
    echo "Error: HF_REPO is required."
    echo "Usage: ./run.sh --hf-repo \"ChapAF/steerable-dolphin-8b\" [--port 8000]"
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

  echo -e "\033[1;32mSubmitting job to SLURM gpuq partition...\033[0m"
  echo "Repo: ${HF_REPO_VAL}"
  echo "Port: ${PORT_VAL}"

  # Re-invoke this exact script using sbatch, passing variables down via --export
  sbatch --export=ALL,HF_REPO="${HF_REPO_VAL}",HF_TOKEN="${HF_TOKEN_VAL}",PORT="${PORT_VAL}",COMPILED_ADAPTER="${COMPILED_ADAPTER_VAL}" "${SCRIPT_PATH}"
  exit 0
fi

# =========================================================================
# RUNNING INSIDE COMPUTE NODE: Set up environment and start application
# =========================================================================
cd "${SLURM_SUBMIT_DIR}"

# Load local cluster dependencies
module purge
module load python/3.10

# Fast check virtual environment setup
VENV_DIR="${SLURM_SUBMIT_DIR}/.venv_slurm"

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating a fresh virtual environment at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip
    pip install --no-cache-dir -r requirements.txt
else
    echo "Virtual environment already exists. Activating..."
    source "${VENV_DIR}/bin/activate"
fi

echo "Starting Steerable Model Runner for ${HF_REPO} on port ${PORT}..."
exec python3 server.py
