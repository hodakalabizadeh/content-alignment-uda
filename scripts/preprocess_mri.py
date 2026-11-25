"""Preprocess the MRI data once and save it, so training and evaluation read arrays instead of NIfTIs.

Offline preprocessing for the 3D configs (obhc.yaml, harp.yaml). For each data split
(train, test), histogram-standardisation landmarks are fitted once on all source and target volumes of the
split (minus excluded subjects); every volume is then histogram-standardised, z-normalised and cropped to the
hippocampal ROI, every label cropped and binarised, and the arrays are saved under
<data.preprocessed_dir>/<split>/ with the landmarks and a meta.json. All other scripts read these arrays
whenever the config sets data.preprocessed_dir.

    python scripts/preprocess_mri.py --config configs/obhc.yaml                 # train and test
    python scripts/preprocess_mri.py --config configs/obhc.yaml --splits train  # one split
    python scripts/preprocess_mri.py --config configs/obhc.yaml --force         # redo even if up to date

Rerunning is cheap: a split whose files and preprocessing are unchanged is left as it is (only missing
arrays are written); if they changed, the split is redone.
"""
import argparse

from ca_core.datasets.mri import preprocess_split
from ca_core.utils import load_config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="MRI experiment config, e.g. configs/obhc.yaml")
    p.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    p.add_argument("--force", action="store_true", help="redo the preprocessing even if it is up to date")
    p.add_argument("--workers", type=int, default=4, help="threads reading and preprocessing the volumes")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if cfg["dataset"] != "mri":
        raise SystemExit(f"{args.config} is not an MRI config (dataset: {cfg['dataset']}); nothing to preprocess")
    if not cfg["data"].get("preprocessed_dir"):
        raise SystemExit(f"{args.config} has no data.preprocessed_dir: set it to where the arrays should go")
    for split in args.splits:
        print(f"[{split}] preprocessed data in {preprocess_split(cfg, split, args.force, args.workers)}", flush=True)


if __name__ == "__main__":
    main()
