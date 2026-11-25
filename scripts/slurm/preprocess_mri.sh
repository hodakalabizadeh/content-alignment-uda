#!/bin/bash
#SBATCH --job-name=preprocess_mri
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Preprocess an MRI config's data once and save it (data.preprocessed_dir); CPU only, no GPU needed.
#
# Usage, from the repository root with the Python environment activated:
#   mkdir -p outputs/logs                    # once: Slurm needs the log folder to exist
#   sbatch scripts/slurm/preprocess_mri.sh <config> [extra flags for scripts/preprocess_mri.py]
# <config> is a file name in configs/ without .yaml (obhc or harp), e.g.
#   sbatch scripts/slurm/preprocess_mri.sh obhc --force
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # so the scripts can import ca_core
python -u scripts/preprocess_mri.py --config "configs/$1.yaml" --workers "${SLURM_CPUS_PER_TASK:-4}" "${@:2}"
