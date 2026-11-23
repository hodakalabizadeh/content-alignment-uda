#!/bin/bash
#SBATCH --job-name=eval_content_alignment
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Evaluate a trained CA run (label propagation on the training data).
#
# Usage, from the repository root with the Python environment activated:
#   mkdir -p outputs/logs                    # once: Slurm needs the log folder to exist
#   sbatch scripts/slurm/eval_content_alignment.sh <config> [extra flags for scripts/eval_content_alignment.py]
# <config> is a file name in configs/ without .yaml (obhc, harp or mnist), e.g.
#   sbatch scripts/slurm/eval_content_alignment.sh harp --run-name ca
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # so the scripts can import ca_core
python -u scripts/eval_content_alignment.py --config "configs/$1.yaml" "${@:2}"
