"""Brain MRI (source, target) pairs for Content Alignment, preprocessed once and held in memory.

Inputs are NIfTI volumes already in 1 mm MNI space, skull-stripped and bias-corrected, with
hippocampus label maps. Each volume is
    1. histogram-standardised over the brain (torchio; landmarks fitted and applied on brain voxels
       only, see `brain_mask`), with the background set back to 0,
    2. z-normalised over the brain (voxels above the volume mean),
    3. cropped to the hippocampal region of interest, (96, 128, 80) voxels.
Labels are cropped the same way and binarised (any structure > 0 is foreground).

Landmarks are fitted on the volumes passed to each dataset (source and target of one split together).
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import torchio as tio
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from ..utils import list_files

# voxels removed from the (start, end) of each axis of the MNI volume -> (96, 128, 80) hippocampal ROI
ROI_CROP = ((51, 50), (53, 52), (35, 74))
SHAPE = (96, 128, 80)  # network input size
CHANNELS = 1
BRAIN_THRESHOLD = 0.10  # brain = voxels above this fraction of the volume's 99th percentile of positive intensities


def _crop(array):
    """Crop to the ROI; a copy, so the full volume is freed."""
    return array[tuple(slice(start, array.shape[ax] - end) for ax, (start, end) in enumerate(ROI_CROP))].copy()


def brain_mask(x):
    """Brain voxels of a volume: above BRAIN_THRESHOLD x its 99th percentile of positive intensities.

    Background is ~80% of an MNI volume, so histogram landmarks fitted on the whole volume are mostly 0.
    A relative threshold (rather than x > 0) also drops the near-zero interpolation rim and negative
    values that some datasets have around the brain.
    """
    return x > BRAIN_THRESHOLD * torch.quantile(x[x > 0].float(), 0.99)


def fit_landmarks(vol_paths):
    """Histogram-standardisation landmarks fitted on the brain voxels of the given volumes."""
    return tio.HistogramStandardization.train([str(p) for p in vol_paths], masking_function=brain_mask)


def load_volume(path, landmarks):
    """Brain-masked histogram standardisation (background -> 0) -> z-normalisation over the brain -> ROI crop.

    Returns float32 (96, 128, 80).
    """
    data = torch.tensor(nib.load(path).get_fdata(), dtype=torch.float32).unsqueeze(0)
    image = tio.HistogramStandardization({'default_image_name': landmarks}, masking_method=brain_mask)(
        tio.ScalarImage(tensor=data))
    image = tio.ScalarImage(tensor=torch.where(brain_mask(data), image.data, torch.zeros_like(image.data)))
    image = tio.ZNormalization(masking_method=lambda x: x > x.mean())(image)
    return _crop(image.numpy()[0])


def load_label(path):
    """ROI-cropped binary label map, float32 (96, 128, 80)."""
    return (_crop(nib.load(path).get_fdata()) > 0).astype(np.float32)


def read_exclusions(path):
    """Subject IDs listed one per line in a text file ('#' starts a comment); empty if the file does not exist."""
    try:
        with open(path) as f:
            return [line.split('#')[0].strip() for line in f if line.split('#')[0].strip()]
    except FileNotFoundError:
        return []


class MRIPairs(Dataset):
    """(source, target) volume pairs with their one-hot labels.

    Args:
        source_vols, source_labels, target_vols, target_labels: lists of NIfTI paths (matched by order).
        pairing: 'random' - one item per target volume, paired with a random source volume (np.random);
                 'all'    - one item per (source, target) combination, sources outer, targets inner.
        exclude: subject IDs; any file whose path contains one of them is dropped.
        workers: threads used to read and preprocess the volumes.

    Items: (source, target, source_label, target_label) float32 tensors, volumes (1, 96, 128, 80)
    and labels one-hot (2, 96, 128, 80).
    """

    def __init__(self, source_vols, source_labels, target_vols, target_labels, pairing='random', exclude=(),
                 workers=4):
        assert pairing in ('random', 'all'), "pairing must be 'random' or 'all'"
        keep = lambda paths: [p for p in paths if not any(s in str(p) for s in exclude)]
        source_vols, source_labels, target_vols, target_labels = map(
            keep, (source_vols, source_labels, target_vols, target_labels))
        assert len(source_vols) == len(source_labels) and len(target_vols) == len(target_labels)

        landmarks = fit_landmarks(source_vols + target_vols)
        with ThreadPoolExecutor(workers) as pool:  # NIfTI reading dominates; results keep the input order
            vol = lambda p: self._volume(load_volume(p, landmarks))
            lab = lambda p: self._one_hot(load_label(p))
            self.source, self.target = list(pool.map(vol, source_vols)), list(pool.map(vol, target_vols))
            self.source_labels, self.target_labels = list(pool.map(lab, source_labels)), list(pool.map(lab, target_labels))
        self.pairing = pairing
        print(f"MRIPairs: {len(self.source)} source and {len(self.target)} target volumes, {len(self)} items")

    @staticmethod
    def _volume(array):
        return torch.from_numpy(array).unsqueeze(0)

    @staticmethod
    def _one_hot(label):
        return F.one_hot(torch.from_numpy(label).long(), num_classes=2).permute(3, 0, 1, 2).float()

    def __len__(self):
        return len(self.target) * (len(self.source) if self.pairing == 'all' else 1)

    def __getitem__(self, index):
        if self.pairing == 'all':
            s, t = divmod(index, len(self.target))
        else:
            s, t = np.random.randint(0, len(self.source)), index
        return self.source[s], self.target[t], self.source_labels[s], self.target_labels[t]


# ---------------- datasets for the experiment scripts (configured by configs/<name>.yaml) ----------------

def _files(cfg, split):
    """(source images, source labels, target images, target labels) file lists of a data split."""
    d = cfg['data'][split]
    keys = ('source_images', 'source_labels', 'target_images', 'target_labels')
    return [np.array(list_files(os.path.join(cfg['data_root'], d[k]))) for k in keys]


def _pairs(cfg, src_vols, src_labels, tgt_vols, tgt_labels):
    """MRIPairs builder for index arrays into the training file lists."""
    return lambda s, t: MRIPairs(list(src_vols[s]), list(src_labels[s]), list(tgt_vols[t]), list(tgt_labels[t]),
                                 pairing=cfg['data']['pairing'])


def ca_fold_split(cfg, folds, fold, split_seed):
    """(train_set, val_set) of one Content Alignment cross-validation fold of the training data.

    Source and target volumes are split separately, with KFold(folds, shuffle=True, random_state=split_seed);
    fold `fold` (0-based) is held out for validation.
    """
    src_vols, src_labels, tgt_vols, tgt_labels = _files(cfg, 'train')
    pairs = _pairs(cfg, src_vols, src_labels, tgt_vols, tgt_labels)
    splits = KFold(n_splits=folds, shuffle=True, random_state=split_seed)
    src_train, src_val = list(splits.split(src_vols))[fold]
    tgt_train, tgt_val = list(splits.split(tgt_vols))[fold]
    return pairs(src_train, tgt_train), pairs(src_val, tgt_val)


def cv_folds(cfg, folds, draw_fold_seed=False):
    """Yield (train_set, val_set) for each cross-validation fold of the training data (downstream segmenter).

    Source and target volumes are split separately (shuffled KFold). With draw_fold_seed, KFold's
    random_state is drawn from the (seeded) NumPy RNG, as in the original downstream training.
    """
    src_vols, src_labels, tgt_vols, tgt_labels = _files(cfg, 'train')
    pairs = _pairs(cfg, src_vols, src_labels, tgt_vols, tgt_labels)
    splits = KFold(n_splits=folds, shuffle=True, random_state=np.random.randint(0, 1000) if draw_fold_seed else None)
    for (src_train, src_val), (tgt_train, tgt_val) in zip(splits.split(src_vols), splits.split(tgt_vols)):
        yield pairs(src_train, tgt_train), pairs(src_val, tgt_val)


def registration_eval_set(cfg):
    """All training pairs: registration is evaluated on the data it was fitted on."""
    return MRIPairs(*map(list, _files(cfg, 'train')), pairing=cfg['data']['pairing'])


def segmentation_test_set(cfg):
    """The held-out test volumes (only the targets are segmented), minus any excluded subjects."""
    exclude_file = cfg['data']['test'].get('exclude_file')
    exclude = read_exclusions(os.path.join(cfg['data_root'], exclude_file)) if exclude_file else []
    return MRIPairs(*map(list, _files(cfg, 'test')), exclude=exclude)
