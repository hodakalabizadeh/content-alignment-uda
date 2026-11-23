"""Small helpers shared by the entry scripts: configs, seeding, listing data files, early stopping and
choosing the CA fold."""
import json
import os
import random

import numpy as np
import torch
import yaml


def load_config(path):
    """Read a YAML experiment config (configs/<name>.yaml) into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(cfg, path):
    """Write the config actually used next to a run's outputs."""
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def set_seed(seed: int) -> None:
    """Seed Python's ``random``, NumPy's global RNG and torch (CPU and all GPUs).

    Call once at the start of an entry script, right after argument parsing.
    This fixes data sampling (random source/target pairing, fold splits,
    DataLoader shuffling) and weight initialisation. It does not make CUDA
    kernels deterministic: the backward pass of ``grid_sample`` (used by the
    spatial transformer) is non-deterministic on GPU, so repeated training
    runs can still differ slightly.

    The data loaders draw from NumPy's global RNG. If a DataLoader is ever
    given ``num_workers > 0``, seed NumPy inside each worker (for example in a
    ``worker_init_fn`` using ``torch.initial_seed()``), otherwise workers
    repeat each other's draws.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_files(folder):
    """Paths of the files directly inside `folder`, sorted (so volumes and labels line up by name)."""
    return sorted(entry.path for entry in os.scandir(folder) if entry.is_file())


def save_ca_fold_record(run_dir, record):
    """Write one CA fold's training record (fold, best validation loss, best epoch, ...) to ca_fold<k>.json."""
    with open(os.path.join(run_dir, f"ca_fold{record['fold']}.json"), 'w') as f:
        json.dump(record, f, indent=1)


def select_ca_fold(run_dir, folds):
    """The CA fold to use downstream: the one with the lowest validation loss (R's loss; no target labels).

    Reads ca_fold<k>.json for k = 0..folds-1 (written by scripts/train_content_alignment.py) and
    returns (fold, records). Fails if any fold has not finished training.
    """
    records = []
    for k in range(folds):
        path = os.path.join(run_dir, f"ca_fold{k}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: CA fold {k} has not finished training")
        with open(path) as f:
            records.append(json.load(f))
    best = min(records, key=lambda r: r['best_val_loss'])
    return best['fold'], records


class EarlyStopping:
    """Save the model whenever the validation loss improves; stop after `patience` epochs without improvement.

    Call once per epoch with the validation loss. `early_stop` becomes True once the loss has not
    improved for `patience` consecutive epochs. Only the model's state_dict is saved, to `path`.
    """

    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None or score >= self.best_score:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
        else:
            self.counter += 1
            print('Early Stopping Counter: ', self.counter, '/', self.patience)
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model, path):
        print('Validation loss decreased: ', self.val_loss_min, ' --> ', val_loss, 'Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
