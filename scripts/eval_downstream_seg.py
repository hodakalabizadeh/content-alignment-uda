"""Evaluate the downstream segmenter (every CV fold) on the held-out target test set.

Each fold's segmenter predicts the target test images (MRI: each hemisphere separately), scored
with Dice, surface Dice, 95% Hausdorff and RAVD, averaged over folds. Prediction runs in fp32.
Reads and writes outputs/<config>/<run-name>/downstream_seed<seed>/ (downstream_eval.csv).

Example:
    python scripts/eval_downstream_seg.py --config configs/harp.yaml --run-name ca --seed 42
"""
import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from ca_core import datasets
from ca_core.evaluation import load_segmenter, predict, results_table, segmentation_metrics, summarise_folds
from ca_core.utils import load_config, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="experiment config, e.g. configs/harp.yaml")
    p.add_argument("--run-name", default="ca")
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42, help="downstream seed of the run to evaluate")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    d = cfg["downstream"]
    set_seed(args.seed)

    data = datasets.load(cfg["dataset"])
    out_dir = Path(args.outputs) / Path(args.config).stem / args.run_name / f"downstream_seed{args.seed}"
    loader = DataLoader(data.segmentation_test_set(cfg), batch_size=d["eval_batch_size"], shuffle=False)

    fold_metrics = []
    print(f"================ {args.config}, run '{args.run_name}', downstream seed {args.seed} ================")
    for fold in range(d["folds"]):
        model = load_segmenter(out_dir / f"downstream_fold{fold}.pt", data.SHAPE, data.CHANNELS)
        pred, true = predict(model, loader, device=args.device, split_hemispheres=d["split_hemispheres"])
        m = segmentation_metrics(true, pred)
        fold_metrics.append(m)
        print(f"Fold {fold}: min DSC = {np.min(m['dsc'])}, mean DSC = {np.mean(m['dsc'])}")

    table = results_table(f"{args.run_name}_seed{args.seed}", summarise_folds(fold_metrics))
    print(table.to_string(index=False))
    table.to_csv(out_dir / "downstream_eval.csv", index=False)
    print(f"\nWrote {out_dir / 'downstream_eval.csv'}")


if __name__ == "__main__":
    main()
