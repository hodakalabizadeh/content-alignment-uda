"""Data for the experiments: one module per data type (mri, mnist).

Each module provides SHAPE, CHANNELS, ca_fold_split(cfg, folds, fold, split_seed) (CA training, one fold),
cv_folds(cfg, folds, draw_fold_seed) (downstream training),
registration_eval_set(cfg) and segmentation_test_set(cfg); `load(name)` returns the module.
"""
from importlib import import_module


def load(name):
    """The data module named by a config's `dataset` field, e.g. load('mri')."""
    return import_module(f"{__name__}.{name}")
