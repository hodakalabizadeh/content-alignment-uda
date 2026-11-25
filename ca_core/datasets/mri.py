"""Brain MRI (source, target) pairs for Content Alignment, preprocessed and held in memory.

Inputs are NIfTI volumes already in 1 mm MNI space, skull-stripped and bias-corrected, with
hippocampus label maps. Each volume is
    1. histogram-standardised over the brain (torchio; landmarks fitted and applied on brain voxels
       only, see `brain_mask`), with the background set back to 0,
    2. z-normalised over the brain (voxels above the volume mean),
    3. cropped to the hippocampal region of interest, (96, 128, 80) voxels.
Labels are cropped the same way and binarised (any structure > 0 is foreground).

Where the preprocessing happens depends on the config's `data.preprocessed_dir`:
  - set: scripts/preprocess_mri.py preprocesses each data split (train, test) once, nnU-Net style, with
    landmarks fitted on all source and target volumes of the split (minus excluded subjects), and saves
    the arrays under <preprocessed_dir>/<split>/. Datasets then read these arrays instead of the NIfTIs.
  - unset: each dataset preprocesses its volumes in memory when it is built, with landmarks fitted on the
    volumes passed to it (source and target of one split, or of one cross-validation fold).
"""
import hashlib
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import torchio as tio
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils import list_files

# voxels removed from the (start, end) of each axis of the MNI volume -> (96, 128, 80) hippocampal ROI
ROI_CROP = ((51, 50), (53, 52), (35, 74))
SHAPE = (96, 128, 80)  # network input size
CHANNELS = 1
BRAIN_THRESHOLD = 0.10  # brain = voxels above this fraction of the volume's 99th percentile of positive intensities
PREPROCESSING_VERSION = 1  # bump whenever the preprocessing changes, so saved arrays are redone


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


def _excluded(paths, exclude):
    """`paths` without those that contain one of the subject IDs in `exclude`."""
    return [p for p in paths if not any(s in str(p) for s in exclude)]


# ---------------- saved preprocessing (data.preprocessed_dir, written by scripts/preprocess_mri.py) ----------------

def _array_path(split_dir, kind, path):
    """Where the preprocessed array of a NIfTI file is saved: <split_dir>/<kind>/<SHA-1 of its absolute path>.npy."""
    return os.path.join(split_dir, kind, hashlib.sha1(os.path.abspath(path).encode()).hexdigest() + '.npy')


def _save(path, array):
    """np.save through a temporary file and a rename, so an interrupted write never leaves a truncated file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.{uuid.uuid4().hex}.tmp'
    with open(tmp, 'wb') as f:
        np.save(f, array)
    os.replace(tmp, path)


def _load_saved(split_dir, kind, path, compute):
    """The saved array of `path`; if it is missing, it is computed with `compute()` and saved first."""
    saved = _array_path(split_dir, kind, path)
    if os.path.exists(saved):
        return np.load(saved)
    array = compute()
    _save(saved, array)
    return array


def _split_files(cfg, split):
    """(volumes, labels) of a data split, source then target, minus the split's excluded subjects."""
    src_vols, src_labels, tgt_vols, tgt_labels = (list(f) for f in _files(cfg, split))
    exclude_file = cfg['data'][split].get('exclude_file')
    exclude = read_exclusions(os.path.join(cfg['data_root'], exclude_file)) if exclude_file else []
    return _excluded(src_vols + tgt_vols, exclude), _excluded(src_labels + tgt_labels, exclude)


def _meta(vols, labels):
    """What a split's saved arrays depend on; a split is redone when this changes."""
    return {'version': PREPROCESSING_VERSION, 'roi_crop': [list(c) for c in ROI_CROP],
            'brain_threshold': BRAIN_THRESHOLD,
            'volumes': sorted(os.path.abspath(p) for p in vols),
            'labels': sorted(os.path.abspath(p) for p in labels)}


def _read_meta(split_dir):
    try:
        with open(os.path.join(split_dir, 'meta.json')) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def preprocess_split(cfg, split, force=False, workers=4):
    """Preprocess one data split ('train' or 'test') and save it under <data.preprocessed_dir>/<split>/.

    Landmarks are fitted once on all source and target volumes of the split (minus excluded subjects); every
    volume and label is then preprocessed and saved as a float32 (96, 128, 80) array, with the landmarks and a
    meta.json. A split that is already up to date (same files and preprocessing) only gets its missing arrays;
    otherwise, or with `force`, it is redone. Returns the split's directory.
    """
    split_dir = os.path.join(cfg['data']['preprocessed_dir'], split)
    vols, labels = _split_files(cfg, split)
    meta = _meta(vols, labels)
    meta_path = os.path.join(split_dir, 'meta.json')
    landmarks_path = os.path.join(split_dir, 'landmarks.npy')

    up_to_date = not force and _read_meta(split_dir) == meta
    if not up_to_date:
        if os.path.exists(meta_path):
            os.remove(meta_path)  # the split is invalid until it has been preprocessed again in full
        print(f'[{split}] fitting histogram landmarks on {len(vols)} volumes', flush=True)
        _save(landmarks_path, np.asarray(fit_landmarks(vols)))
    landmarks = np.load(landmarks_path)

    for kind, paths, compute in (('volumes', vols, lambda p: load_volume(p, landmarks)),
                                 ('labels', labels, load_label)):
        todo = [p for p in paths if not up_to_date or not os.path.exists(_array_path(split_dir, kind, p))]
        with ThreadPoolExecutor(workers) as pool:  # NIfTI reading dominates
            list(tqdm(pool.map(lambda p: _save(_array_path(split_dir, kind, p), compute(p)), todo),
                      total=len(todo), desc=f'[{split}] {kind}'))

    with open(meta_path, 'w') as f:  # written last: marks the split as complete
        json.dump(meta, f, indent=1)
    return split_dir


def _preprocessed(cfg, split):
    """The split's saved preprocessing directory, checked against the config; None if data.preprocessed_dir is unset."""
    if not cfg['data'].get('preprocessed_dir'):
        return None
    split_dir = os.path.join(cfg['data']['preprocessed_dir'], split)
    meta = _read_meta(split_dir)
    if meta is None:
        raise FileNotFoundError(f"no preprocessed '{split}' data in {split_dir}: run "
                                f"`python scripts/preprocess_mri.py --config <this config>` first, "
                                f"or remove data.preprocessed_dir from the config")
    if meta != _meta(*_split_files(cfg, split)):
        raise RuntimeError(f"the preprocessed '{split}' data in {split_dir} does not match the config's files or the "
                           f"preprocessing: rerun `python scripts/preprocess_mri.py --config <this config>`")
    return split_dir


class MRIPairs(Dataset):
    """(source, target) volume pairs with their one-hot labels.

    Args:
        source_vols, source_labels, target_vols, target_labels: lists of NIfTI paths (matched by order).
        pairing: 'random' - one item per target volume, paired with a random source volume (np.random);
                 'all'    - one item per (source, target) combination, sources outer, targets inner.
        exclude: subject IDs; any file whose path contains one of them is dropped.
        workers: threads used to read and preprocess the volumes.
        preprocessed: a split directory written by preprocess_split; the volumes and labels are read from it
                      (preprocessed with the split's landmarks) instead of being preprocessed here.

    Items: (source, target, source_label, target_label) float32 tensors, volumes (1, 96, 128, 80)
    and labels one-hot (2, 96, 128, 80).
    """

    def __init__(self, source_vols, source_labels, target_vols, target_labels, pairing='random', exclude=(),
                 workers=4, preprocessed=None):
        assert pairing in ('random', 'all'), "pairing must be 'random' or 'all'"
        source_vols, source_labels, target_vols, target_labels = (
            _excluded(paths, exclude) for paths in (source_vols, source_labels, target_vols, target_labels))
        assert len(source_vols) == len(source_labels) and len(target_vols) == len(target_labels)

        if preprocessed is None:
            landmarks = fit_landmarks(source_vols + target_vols)
            volume, label = (lambda p: load_volume(p, landmarks)), load_label
        else:
            landmarks = np.load(os.path.join(preprocessed, 'landmarks.npy'))
            volume = lambda p: _load_saved(preprocessed, 'volumes', p, lambda: load_volume(p, landmarks))
            label = lambda p: _load_saved(preprocessed, 'labels', p, lambda: load_label(p))
        with ThreadPoolExecutor(workers) as pool:  # NIfTI reading dominates; results keep the input order
            vol = lambda p: self._volume(volume(p))
            lab = lambda p: self._one_hot(label(p))
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
    preprocessed = _preprocessed(cfg, 'train')
    return lambda s, t: MRIPairs(list(src_vols[s]), list(src_labels[s]), list(tgt_vols[t]), list(tgt_labels[t]),
                                 pairing=cfg['data']['pairing'], preprocessed=preprocessed)


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
    return MRIPairs(*map(list, _files(cfg, 'train')), pairing=cfg['data']['pairing'],
                    preprocessed=_preprocessed(cfg, 'train'))


def segmentation_test_set(cfg):
    """The held-out test volumes (only the targets are segmented), minus any excluded subjects."""
    exclude_file = cfg['data']['test'].get('exclude_file')
    exclude = read_exclusions(os.path.join(cfg['data_root'], exclude_file)) if exclude_file else []
    return MRIPairs(*map(list, _files(cfg, 'test')), exclude=exclude, preprocessed=_preprocessed(cfg, 'test'))
