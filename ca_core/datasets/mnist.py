"""Morpho-MNIST digits rendered MNIST-M style, for the 2D Content Alignment experiments.

A digit image becomes a colour image: pixels of the digit (intensity >= threshold) take their
colour from one CIFAR-10 patch (15x15, upsampled to 32x32), the background from another (5x5,
upsampled), and Gaussian noise is added. Source and target domains differ in digit thickness
(e.g. Morpho-MNIST 'thic' -> 'thin'); the style (CIFAR patches + noise) is drawn per item.
All randomness uses np.random, so set_seed makes it reproducible.
"""
import gzip
import struct

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from torchvision import datasets

THRESHOLD = 0.3  # digit intensity (in [0, 1]) above which a pixel belongs to the mask
SHAPE = (32, 32)  # network input size
CHANNELS = 3


def load_idx(path):
    """Read a uint8 array stored in MNIST's IDX format (gzip-compressed if the path ends in '.gz').

    Adapted from Morpho-MNIST (https://github.com/dccastro/Morpho-MNIST),
    MIT License, Copyright (c) 2021 Daniel Coelho de Castro.
    """
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rb') as f:
        ndim = struct.unpack('BBBB', f.read(4))[3]
        shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(int(np.prod(shape))), dtype=np.uint8).reshape(shape)


def load_morpho_mnist(data_dir, morph, digit, split='train'):
    """One digit class of a Morpho-MNIST style, scaled to [0, 1] and zero-padded to 32x32, with its masks.

    split: 'train' or 't10k' (test). Returns (images (N, 32, 32) float, masks (N, 32, 32) int).
    """
    x = load_idx(f"{data_dir}/{morph}/{split}-images-idx3-ubyte.gz")
    y = load_idx(f"{data_dir}/{morph}/{split}-labels-idx1-ubyte.gz")
    x = np.pad(x[y == digit] / 255, ((0, 0), (2, 2), (2, 2)), 'constant')
    return x, (x > THRESHOLD).astype(int)


def random_patch(cifar_image, patch_size, output_size=(32, 32)):
    """A random patch_size crop of a CIFAR image, resized to output_size."""
    x = np.random.randint(0, cifar_image.shape[0] - patch_size[0])
    y = np.random.randint(0, cifar_image.shape[1] - patch_size[1])
    return cv2.resize(cifar_image[x:x + patch_size[0], y:y + patch_size[1]], output_size)


def mnist_m(digit, foreground, background, threshold=THRESHOLD, noise_std=0.05):
    """Fill the digit with `foreground`, the rest with `background`, add Gaussian noise. Returns (3, H, W)."""
    image = np.where((digit >= threshold)[..., None], foreground, background)
    if noise_std:
        noise = np.random.normal(0, noise_std, image.shape)
        image = np.clip(image + noise, np.min(image), np.max(image))  # stay within the noiseless range
    return image.transpose(2, 0, 1)


def one_hot(mask):
    """(H, W) integer mask -> (2, H, W) float one-hot tensor."""
    return F.one_hot(torch.as_tensor(mask).long(), num_classes=2).permute(2, 0, 1).float()


class MNISTMPairs(Dataset):
    """(source, target) MNIST-M pairs for registration / CA training.

    Item i pairs source digit i with a random target digit; both are rendered with the SAME pair of
    CIFAR patches (a random CIFAR image for the background and the next one for the foreground) but
    independent noise, so they differ in digit shape, not in style.

    Items: (source, target, source_label, target_label) float32 tensors, images (3, 32, 32) and
    labels one-hot (2, 32, 32).
    """

    def __init__(self, source, source_masks, target, target_masks, cifar, threshold=THRESHOLD, noise_std=0.05,
                 patch_size_bg=(5, 5), patch_size_fg=(15, 15)):
        self.source, self.source_masks = source, source_masks
        self.target, self.target_masks = target, target_masks
        self.cifar = cifar
        self.threshold, self.noise_std = threshold, noise_std
        self.patch_size_bg, self.patch_size_fg = patch_size_bg, patch_size_fg

    def __len__(self):
        return min(len(self.source), len(self.target))

    def __getitem__(self, index):
        # note: randint's upper bound excludes the last target/CIFAR image; kept as trained
        t = np.random.randint(0, len(self.target) - 1)
        bg = np.random.randint(0, len(self.cifar) - 1)
        background = random_patch(self.cifar[bg], self.patch_size_bg)
        foreground = random_patch(self.cifar[(bg + 1) % len(self.cifar)], self.patch_size_fg)
        render = lambda digit: torch.from_numpy(
            mnist_m(digit, foreground, background, self.threshold, self.noise_std)).float()
        return (render(self.source[index]), render(self.target[t]),
                one_hot(self.source_masks[index]), one_hot(self.target_masks[t]))


class MNISTMImages(Dataset):
    """MNIST-M images with their masks, for evaluating a segmenter.

    Image i uses CIFAR image i for the background patch and image i + 1 for the foreground patch.
    Items: (image (3, 32, 32), one-hot label (2, 32, 32)) float32 tensors.
    """

    def __init__(self, images, masks, cifar, threshold=THRESHOLD, noise_std=0.05,
                 patch_size_bg=(5, 5), patch_size_fg=(15, 15)):
        self.images, self.masks, self.cifar = images, masks, cifar
        self.threshold, self.noise_std = threshold, noise_std
        self.patch_size_bg, self.patch_size_fg = patch_size_bg, patch_size_fg

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        background = random_patch(self.cifar[index], self.patch_size_bg)
        foreground = random_patch(self.cifar[index + 1], self.patch_size_fg)
        image = mnist_m(self.images[index], foreground, background, self.threshold, self.noise_std)
        return torch.from_numpy(image).float(), one_hot(self.masks[index])


# ---------------- datasets for the experiment scripts (configured by configs/mnist.yaml) ----------------

def _cifar(cfg, train):
    """CIFAR-10 images scaled to [0, 1], (N, 32, 32, 3)."""
    return datasets.CIFAR10(root=cfg['data_root'], train=train, download=True).data / 255


def _train_digits(cfg):
    d = cfg['data']
    return (*load_morpho_mnist(cfg['data_root'], d['source_style'], d['digit']),
            *load_morpho_mnist(cfg['data_root'], d['target_style'], d['digit']))


def ca_fold_split(cfg, folds, fold, split_seed):
    """(train_set, val_set) of one Content Alignment cross-validation fold of the training digits.

    KFold(folds, shuffle=True, random_state=split_seed) over the source digits, fold `fold` (0-based) held out for
    validation; the target digits use the same indices.
    """
    xs, ys, xt, yt = _train_digits(cfg)
    cifar = _cifar(cfg, train=True)
    pairs = lambda i: MNISTMPairs(xs[i], ys[i], xt[i], yt[i], cifar, noise_std=cfg['data']['noise_std'])
    train_idx, val_idx = list(KFold(n_splits=folds, shuffle=True, random_state=split_seed).split(xs))[fold]
    return pairs(train_idx), pairs(val_idx)


def cv_folds(cfg, folds, draw_fold_seed=False):
    """Yield (train_set, val_set) for each cross-validation fold of the training digits (downstream segmenter).

    One shuffled KFold split of the source digits; the target digits use the same indices.
    """
    xs, ys, xt, yt = _train_digits(cfg)
    cifar = _cifar(cfg, train=True)
    splits = KFold(n_splits=folds, shuffle=True, random_state=np.random.randint(0, 1000) if draw_fold_seed else None)
    pairs = lambda i: MNISTMPairs(xs[i], ys[i], xt[i], yt[i], cifar, noise_std=cfg['data']['noise_std'])
    for train_idx, val_idx in splits.split(xs):
        yield pairs(train_idx), pairs(val_idx)


def registration_eval_set(cfg):
    """All training pairs: registration is evaluated on the digits it was fitted on."""
    return MNISTMPairs(*_train_digits(cfg), _cifar(cfg, train=True), noise_std=cfg['data']['noise_std'])


def segmentation_test_set(cfg):
    """Held-out target-style test digits, rendered on CIFAR-10 test images."""
    d = cfg['data']
    images, masks = load_morpho_mnist(cfg['data_root'], d['target_style'], d['digit'], split='t10k')
    return MNISTMImages(images, masks, _cifar(cfg, train=False), noise_std=d['noise_std'])
