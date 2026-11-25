"""Train the downstream segmenter on CA-aligned source data, k-fold CV.

The CA model is the fold of the run's CA cross-validation with the lowest validation loss (R's loss, no
target labels; ca_core.utils.select_ca_fold), so all CA folds must have finished training. Each source
image and label is warped to its target by that model, and the segmenter is trained on the warped pairs,
with k-fold CV (`downstream.folds`). Outputs go to outputs/<config>/<run-name>/downstream_seed<seed>/:
downstream_fold{k}.pt, the config used and ca_selection.json (which CA fold was used, and why).

Example:
    python scripts/train_downstream_seg.py --config configs/harp.yaml --run-name ca --seed 42
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ca_core import datasets
from ca_core.training.content_alignment import build_registration
from ca_core.training.downstream_segmentation import build_segmenter, train_segmenter_epoch, validate_segmenter
from ca_core.training.losses import DiceLoss
from ca_core.utils import EarlyStopping, load_config, save_config, select_ca_fold, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="experiment config, e.g. configs/harp.yaml")
    p.add_argument("--run-name", default="ca", help="run directory holding the trained CA folds")
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--epochs", type=int, help="override the maximum number of epochs")
    p.add_argument("--seed", type=int, default=42, help="downstream seed; outputs go to downstream_seed<seed>/")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    d = cfg["downstream"]
    d["epochs"] = args.epochs or d["epochs"]
    set_seed(args.seed)

    data = datasets.load(cfg["dataset"])
    run_dir = Path(args.outputs) / Path(args.config).stem / args.run_name
    ca_fold, records = select_ca_fold(run_dir, cfg["ca"]["folds"])
    ca_checkpoint = run_dir / f"ca_fold{ca_fold}.pt"
    out_dir = run_dir / f"downstream_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "downstream_config.yaml")
    with open(out_dir / "ca_selection.json", "w") as f:
        json.dump(dict(selected_fold=ca_fold, rule="lowest CA validation loss",
                       val_loss={r["fold"]: r["best_val_loss"] for r in records}), f, indent=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = ", ".join(f"fold {r['fold']} {r['best_val_loss']:.4f}" for r in records)
    print(f"Downstream segmentation: {args.config}, run '{args.run_name}', seed {args.seed}, device {device}\n"
          f"CA validation losses: {losses} -> using CA fold {ca_fold} ({ca_checkpoint})")

    for fold, (train_set, val_set) in enumerate(data.cv_folds(cfg, d["folds"], draw_fold_seed=d["draw_fold_seed"])):
        print(f"================ fold {fold}: {len(train_set)} training / {len(val_set)} validation pairs ================")
        train_loader = DataLoader(train_set, batch_size=d["batch_size"], shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=d["batch_size"], shuffle=True, drop_last=d["val_drop_last"],
                                pin_memory=True)
        reg = build_registration(data.SHAPE, data.CHANNELS, cfg["ca"]["int_steps"])
        reg.load_state_dict(torch.load(ca_checkpoint, map_location=device, weights_only=True))
        seg = build_segmenter(data.SHAPE, data.CHANNELS)
        optimiser = torch.optim.AdamW(seg.parameters(), lr=d["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5,
                                                               patience=d["lr_scheduler_patience"])
        early_stopping = EarlyStopping(patience=d["patience"])
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(d["epochs"]):
            train_segmenter_epoch(seg, reg, train_loader, optimiser, DiceLoss(), scaler, device,
                                  split_hemispheres=d["split_hemispheres"], epoch=epoch)
            metrics = validate_segmenter(seg, reg, val_loader, DiceLoss(), device,
                                         split_hemispheres=d["split_hemispheres"], amp=d["val_amp"])
            scheduler.step(metrics["loss"])
            early_stopping(metrics["loss"], seg, out_dir / f"downstream_fold{fold}.pt")
            if early_stopping.early_stop:
                print("Early stopping: no improvement for", d["patience"], "epochs")
                break


if __name__ == "__main__":
    main()
