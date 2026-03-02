#!/usr/bin/env bash

# Usage:
#   source /projects/zaijia001/FoundationPose/source_foundationpose_env.sh

set -eo pipefail

_is_sourced=0
if [[ -n "${ZSH_EVAL_CONTEXT:-}" ]]; then
  case "${ZSH_EVAL_CONTEXT}" in
    *:file) _is_sourced=1 ;;
  esac
elif [[ -n "${BASH_VERSION:-}" ]]; then
  if [[ "${BASH_SOURCE[0]:-}" != "${0}" ]]; then
    _is_sourced=1
  fi
fi

if [[ "${_is_sourced}" -ne 1 ]]; then
  echo "Please source this script instead of executing it:"
  echo "  source /projects/zaijia001/FoundationPose/source_foundationpose_env.sh"
  exit 1
fi

if ! command -v module >/dev/null 2>&1; then
  source /cluster/apps/software/lmod/lmod/init/bash >/dev/null 2>&1 || true
fi
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh >/dev/null 2>&1 || true

module purge
module load CUDA/11.8.0
module load GCCcore/13.2.0
module load libglvnd/1.7.0

conda activate /projects/zaijia001/conda_envs/foundationpose
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

echo "FoundationPose env ready"
echo "python: $(command -v python)"
python -c "import sys, torch; print('sys.executable=', sys.executable); print('torch.cuda.is_available=', torch.cuda.is_available())"
