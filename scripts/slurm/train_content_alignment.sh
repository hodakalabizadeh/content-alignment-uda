#!/bin/bash
#SBATCH --job-name=train_content_alignment
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Train the CA model (R + D + S) with k-fold CV (ca.folds): one fold with --fold k, else all folds in turn.
#
# Usage, from the repository root with the Python environment activated:
#   mkdir -p outputs/logs                    # once: Slurm needs the log folder to exist
#   sbatch scripts/slurm/train_content_alignment.sh <config> [extra flags for scripts/train_content_alignment.py]
# <config> is a file name in configs/ without .yaml (obhc, harp or mnist), e.g.
#   sbatch scripts/slurm/train_content_alignment.sh obhc --run-name ca --fold 0 --seed 42
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # so the scripts can import ca_core
python -u scripts/train_content_alignment.py --config "configs/$1.yaml" "${@:2}"
