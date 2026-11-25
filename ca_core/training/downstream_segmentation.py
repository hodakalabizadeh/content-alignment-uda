"""One epoch of downstream segmenter training and validation on Content Alignment (CA) output.

Each (source, target) pair is passed through a trained, frozen CA registration network; the
segmenter is trained on the source image warped to the target, with the source label warped
along with it. The loader yields (source, target, source_label, target_label) batches.

split_hemispheres=True (MRI) cuts each volume in half along the first spatial axis, runs the
segmenter on each half and sums the two losses. Validation then also reports Dice on the real
TARGET images (one per half); it is logged only and never used for model selection.
"""
import numpy as np
import surface_distance as s_d
import torch
from torch.amp import autocast
from tqdm import tqdm

from ..models.segmentation import SegmentationUNet
from .content_alignment import SEG_FEATURES


def build_segmenter(shape, channels):
    """The downstream segmenter (same U-Net as CA's S); `shape` only sets 2D vs 3D."""
    return SegmentationUNet(in_shape=shape, infeatures=channels, outfeatures=2, nb_features=SEG_FEATURES, max_pool=2)


def _halves(x):
    """Split along the first spatial axis: (left, right)."""
    mid = x.shape[2] // 2
    return x[..., :mid, :, :], x[..., mid:, :, :]


def _aligned_batch(batch, reg_model, device, amp):
    """Source image and label warped to the target by the frozen CA model, plus the target image and label."""
    source, target, source_label, target_label = (t.to(device).float() for t in batch)
    with torch.no_grad(), autocast('cuda', enabled=amp):  # eval mode: the label is warped with nearest neighbour
        source_to_target_img, _, source_to_target_label, _, _ = reg_model(source, target, source_label)
    return source_to_target_img, source_to_target_label, target, target_label


def _segment(seg_model, loss_func, img, label, split_hemispheres):
    """Prediction(s) and loss; with split_hemispheres, preds are (left, right) and the loss is their sum."""
    if not split_hemispheres:
        pred = seg_model(img)
        return pred, loss_func(pred, label)
    (img_l, img_r), (lab_l, lab_r) = _halves(img), _halves(label)
    pred_l, pred_r = seg_model(img_l), seg_model(img_r)
    return (pred_l, pred_r), loss_func(pred_l, lab_l) + loss_func(pred_r, lab_r)


def _dice(pred, label):
    """Mean per-sample Dice of the foreground channel (pred thresholded at 0.5)."""
    pred = (pred >= 0.5).cpu().numpy()
    label = label.cpu().numpy().astype(bool)
    return np.mean([s_d.compute_dice_coefficient(pred[i, 1], label[i, 1]) for i in range(pred.shape[0])])


def train_segmenter_epoch(seg_model, reg_model, dataloader, optim, loss_func, scaler, device,
                          split_hemispheres=False, epoch=None):
    """Train the segmenter for one epoch on CA-aligned source data; returns the mean training loss.

    reg_model: the trained CA registration network (kept frozen).
    Mixed precision follows `scaler` (GradScaler(enabled=False) trains in fp32).
    """
    amp = scaler.is_enabled()
    seg_model.to(device).train()
    reg_model.to(device).eval()

    total_loss, batches = 0.0, 0
    desc = "Training" if epoch is None else f"Epoch {epoch}"
    for batch in tqdm(dataloader, desc=desc, leave=True):
        batches += 1
        img, label, _, _ = _aligned_batch(batch, reg_model, device, amp)

        optim.zero_grad()
        with autocast('cuda', enabled=amp):
            _, loss = _segment(seg_model, loss_func, img, label, split_hemispheres)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        total_loss += loss.item()

    av_loss = total_loss / batches
    print(f'\nTraining set: Average loss: {av_loss:.4f}', flush=True)
    return av_loss


def validate_segmenter(seg_model, reg_model, dataloader, loss_func, device, split_hemispheres=False, amp=False):
    """Validate the segmenter for one epoch on CA-aligned source data.

    Returns a dict of means over the validation set:
        'loss'                     loss on the aligned data (used for early stopping)
        'dice_left', 'dice_right'  with split_hemispheres: Dice on the target images (reported only)
    """
    seg_model.to(device).eval()
    reg_model.to(device).eval()

    totals = dict(loss=0.0, dice_left=0.0, dice_right=0.0)
    batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batches += 1
            img, label, target, target_label = _aligned_batch(batch, reg_model, device, amp)
            with autocast('cuda', enabled=amp):
                pred, loss = _segment(seg_model, loss_func, img, label, split_hemispheres)
                totals['loss'] += loss.item()

                if split_hemispheres:  # Dice on the real target, per hemisphere
                    target_pred = [seg_model(h) for h in _halves(target)]
                    target_label_halves = _halves(target_label)
                    totals['dice_left'] += _dice(target_pred[0], target_label_halves[0])
                    totals['dice_right'] += _dice(target_pred[1], target_label_halves[1])

    metrics = {k: v / batches for k, v in totals.items()}
    if not split_hemispheres:
        metrics = {'loss': metrics['loss']}
    print(f"Validation set: Average loss: {metrics['loss']:.4f}\n", flush=True)

    return metrics
