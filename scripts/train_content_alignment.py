"""Train the Content Alignment model (registration R + discriminator D + segmenter S), with k-fold CV.

The training data are split into `ca.folds` folds (KFold, shuffled with --seed); for each fold a CA model is
trained on the other folds and early-stopped on R's validation loss on the held-out fold (no target labels).
Saves, per fold k, R to outputs/<config>/<run-name>/ca_fold<k>.pt (its best epoch) and ca_fold<k>.json
(best validation loss and epoch), plus the config used. The downstream stage uses the fold with the lowest
validation loss (see ca_core.utils.select_ca_fold).

Each fold is independent: --fold k trains only fold k (e.g. one Slurm job per fold); without --fold all folds
are trained one after another. Every fold starts from the same seed.

Examples:
    python scripts/train_content_alignment.py --config configs/harp.yaml                      # all folds
    python scripts/train_content_alignment.py --config configs/harp.yaml --fold 0 --seed 43   # one fold
    python scripts/train_content_alignment.py --config configs/harp.yaml --run-name ca_no-disc --lambda-disc 0
    python scripts/train_content_alignment.py --config configs/harp.yaml --run-name ca_no-seg --lambda-seg 0
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ca_core import datasets
from ca_core.training.content_alignment import build_ca_models, train_ca_epoch, validate_ca
from ca_core.training.losses import DiceLoss
from ca_core.utils import EarlyStopping, load_config, save_ca_fold_record, save_config, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="experiment config, e.g. configs/harp.yaml")
    p.add_argument("--run-name", default="ca", help="outputs go to <outputs>/<config name>/<run name>/")
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--fold", type=int, help="train only this CA fold (0-based); default: all folds in turn")
    p.add_argument("--lambda-disc", type=float, help="override the discriminator weight (0 disables D)")
    p.add_argument("--lambda-seg", type=float, help="override the segmentation weight (0 disables S)")
    p.add_argument("--epochs", type=int, help="override the maximum number of epochs")
    p.add_argument("--seed", type=int, default=42, help="seeds training and the fold split")
    return p.parse_args()


def train_fold(cfg, data, fold, seed, run_dir, device):
    """Train the CA model of one fold; save its best R and its record. Returns the record."""
    c = cfg["ca"]
    lam = c["lambda"]
    loss_weights = (lam["sim"], lam["smooth"], lam["disc"], lam["seg"])
    set_seed(seed)  # every fold starts from the same seed, so a fold trained alone matches one trained in turn

    reg, disc, seg = build_ca_models(data.SHAPE, data.CHANNELS, c["int_steps"])
    train_set, val_set = data.ca_fold_split(cfg, c["folds"], fold, split_seed=seed)
    print(f"================ CA fold {fold}/{c['folds']}: {len(train_set)} training / {len(val_set)} validation pairs "
          f"================", flush=True)
    reg.to(device)
    opt_reg = torch.optim.Adam(reg.parameters(), lr=c["lr"]["reg"], **({"fused": True} if c["fused_adam"] else {}))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=c["lr"]["disc"], betas=(0.5, 0.999))
    opt_seg = torch.optim.Adam(seg.parameters(), lr=c["lr"]["seg"])
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt_reg, mode="min", factor=0.5,
                                                            patience=c["lr_scheduler_patience"])
                 if c["lr_scheduler_patience"] else None)
    train_loader = DataLoader(train_set, batch_size=c["batch_size"], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=c["batch_size"], shuffle=True, drop_last=True, pin_memory=True)
    early_stopping = EarlyStopping(patience=c["patience"])
    scaler = torch.cuda.amp.GradScaler(enabled=c["amp"])

    best_epoch, epoch = 0, 0
    for epoch in range(c["epochs"]):
        print(f"epoch {epoch}")
        train_ca_epoch(reg, disc, seg, train_loader, opt_reg, opt_disc, opt_seg, nn.BCELoss(), DiceLoss(),
                       loss_weights, scaler, device)
        metrics = validate_ca(reg, disc, seg, val_loader, nn.BCELoss(), DiceLoss(), loss_weights, device,
                              amp=c["amp"])
        if scheduler:
            scheduler.step(metrics["total_loss"])
        early_stopping(metrics["total_loss"], reg, run_dir / f"ca_fold{fold}.pt")
        if early_stopping.counter == 0:
            best_epoch = epoch
        if early_stopping.early_stop:
            print("Early stopping: no improvement for", c["patience"], "epochs")
            break

    record = dict(fold=fold, folds=c["folds"], seed=seed, best_val_loss=float(early_stopping.val_loss_min),
                  best_epoch=best_epoch, epochs_trained=epoch + 1,
                  train_pairs=len(train_set), val_pairs=len(val_set))
    save_ca_fold_record(run_dir, record)
    print(f"CA fold {fold}: best validation loss {record['best_val_loss']:.4f} at epoch {best_epoch}", flush=True)
    return record


def main():
    args = parse_args()
    cfg = load_config(args.config)
    c = cfg["ca"]
    for key, value in [("disc", args.lambda_disc), ("seg", args.lambda_seg)]:
        if value is not None:
            c["lambda"][key] = value
    c["epochs"] = args.epochs or c["epochs"]
    if args.fold is not None and not 0 <= args.fold < c["folds"]:
        raise ValueError(f"--fold must be in 0..{c['folds'] - 1}")

    data = datasets.load(cfg["dataset"])
    run_dir = Path(args.outputs) / Path(args.config).stem / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "ca_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Content Alignment: {args.config}, run '{args.run_name}', seed {args.seed}, lambdas {c['lambda']}, "
          f"device {device}")

    for fold in ([args.fold] if args.fold is not None else range(c["folds"])):
        train_fold(cfg, data, fold, args.seed, run_dir, device)


if __name__ == "__main__":
    main()
