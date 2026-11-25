"""Evaluate the trained CA registration models (every fold) by how well they propagate labels.

Each source label is warped onto its target with a fold's CA model and compared with the target label:
Dice, surface Dice, 95% Hausdorff and RAVD, mean and std across pairs. Registration is evaluated on the
training data it was fitted on; the held-out test data are reserved for the downstream segmentation
(scripts/eval_downstream_seg.py). This uses target labels, so it is for reporting only: the fold used
downstream is chosen by validation loss (ca_core.utils.select_ca_fold), shown in the 'Selected' column.
Writes outputs/<config>/<run-name>/ca_eval.csv (one row per fold).

Example:
    python scripts/eval_content_alignment.py --config configs/harp.yaml --run-name ca
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ca_core import datasets
from ca_core.evaluation import register, results_table, segmentation_metrics, summarise_subjects
from ca_core.training.content_alignment import build_registration
from ca_core.utils import load_config, select_ca_fold, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="experiment config, e.g. configs/harp.yaml")
    p.add_argument("--run-name", default="ca")
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    c = cfg["ca"]
    set_seed(args.seed)

    data = datasets.load(cfg["dataset"])
    run_dir = Path(args.outputs) / Path(args.config).stem / args.run_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected, records = select_ca_fold(run_dir, c["folds"])

    dataset = data.registration_eval_set(cfg)
    loader = DataLoader(dataset, batch_size=c["eval_batch_size"] or len(dataset), shuffle=False)
    model = build_registration(data.SHAPE, data.CHANNELS, c["int_steps"])

    print(f"================ {args.config}, run '{args.run_name}' ================")
    rows = []
    for record in records:
        fold = record["fold"]
        np.random.seed(args.seed)  # the same random source/target pairing for every fold
        model.load_state_dict(torch.load(run_dir / f"ca_fold{fold}.pt", map_location=device, weights_only=True))
        target_labels, warped_labels = register(model, loader, device, amp=c["amp"])
        m = segmentation_metrics(target_labels, warped_labels)
        print(f"Fold {fold}: validation loss {record['best_val_loss']:.4f}, min DSC = {np.min(m['dsc'])}, "
              f"mean DSC = {np.mean(m['dsc'])}")
        row = results_table(f"{args.run_name}_fold{fold}", summarise_subjects(m))
        row.insert(1, "Fold", fold)
        row.insert(2, "Val loss", round(record["best_val_loss"], 4))
        row.insert(3, "Selected", fold == selected)
        rows.append(row)

    table = pd.concat(rows, ignore_index=True)
    print(table.to_string(index=False))
    table.to_csv(run_dir / "ca_eval.csv", index=False)
    print(f"\nWrote {run_dir / 'ca_eval.csv'}")


if __name__ == "__main__":
    main()
