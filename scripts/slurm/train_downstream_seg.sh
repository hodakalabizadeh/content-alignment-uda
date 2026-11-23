#!/bin/bash
#SBATCH --job-name=train_downstream_seg
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Train the downstream segmenter on CA-aligned data (needs a trained CA run).
#
# Usage, from the repository root with the Python environment activated:
#   mkdir -p outputs/logs                    # once: Slurm needs the log folder to exist
#   sbatch scripts/slurm/train_downstream_seg.sh <config> [extra flags for scripts/train_downstream_seg.py]
# <config> is a file name in configs/ without .yaml (obhc, harp or mnist), e.g.
#   sbatch scripts/slurm/train_downstream_seg.sh obhc --run-name ca
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # so the scripts can import ca_core
python -u scripts/train_downstream_seg.py --config "configs/$1.yaml" "${@:2}"
