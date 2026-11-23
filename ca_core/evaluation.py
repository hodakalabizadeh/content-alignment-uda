"""Evaluation: per-subject segmentation metrics, fold summaries, and prediction with the per-fold segmenters.

Metrics (per subject, on the foreground channel, masks thresholded at 0.5):
    dsc          Dice similarity coefficient
    surface_dsc  surface Dice at a tolerance of `tolerance` (default 1.01 voxel units)
    hd95         95th-percentile Hausdorff distance
    ravd         absolute relative volume difference, in %
Distances assume isotropic voxels of size `spacing` (1 mm for the MNI-space MRI).
"""
import numpy as np
import pandas as pd
import surface_distance as s_d
import torch
from torch.cuda.amp import autocast

from .models.segmentation import SegmentationUNet

METRICS = ('dsc', 'surface_dsc', 'hd95', 'ravd')


def segmentation_metrics(true, pred, spacing=1.0, tolerance=1.01):
    """Per-subject metrics for one-hot labels / softmax predictions of shape (N, n_classes, *spatial).

    Accepts torch tensors (on the CPU) or numpy arrays; the inputs are not modified.
    Returns a dict of numpy arrays of length N, keyed by METRICS.
    """
    out = {k: [] for k in METRICS}
    for i in range(true.shape[0]):
        mask_gt = np.asarray(true[i, 1]) >= 0.5
        mask_pred = np.asarray(pred[i, 1]) >= 0.5

        pred_vol, true_vol = np.sum(mask_pred), np.sum(mask_gt)
        out['ravd'].append(abs(pred_vol - true_vol) / float(true_vol) * 100)
        out['dsc'].append(s_d.compute_dice_coefficient(mask_gt, mask_pred))
        surface_dist = s_d.compute_surface_distances(mask_gt, mask_pred, [spacing] * mask_gt.ndim)
        out['hd95'].append(s_d.compute_robust_hausdorff(surface_dist, 95))
        out['surface_dsc'].append(s_d.compute_surface_dice_at_tolerance(surface_dist, tolerance))
    return {k: np.array(v) for k, v in out.items()}


def summarise_folds(fold_metrics):
    """Combine per-fold results: each metric is averaged over subjects within a fold, then the
    fold averages are summarised as mean and std across folds. 'min_dsc' is the lowest subject
    Dice of each fold, averaged across folds.

    fold_metrics: list (one per fold) of dicts returned by segmentation_metrics.
    Returns {metric: (mean, std)} for METRICS, plus 'min_dsc': (mean, std).
    """
    summary = {}
    for k in METRICS:
        fold_means = [np.mean(m[k]) for m in fold_metrics]
        summary[k] = (np.mean(fold_means), np.std(fold_means))
    fold_mins = [np.min(m['dsc']) for m in fold_metrics]
    summary['min_dsc'] = (np.mean(fold_mins), np.std(fold_mins))
    return summary


def summarise_subjects(metrics):
    """Summarise one model's results: each metric as mean and std across subjects; 'min_dsc' is the lowest
    subject Dice (std 0). metrics: dict returned by segmentation_metrics."""
    summary = {k: (np.mean(metrics[k]), np.std(metrics[k])) for k in METRICS}
    summary['min_dsc'] = (np.min(metrics['dsc']), 0.0)
    return summary


def results_table(name, summary):
    """One-row DataFrame with 'mean±std' strings (3 decimals), ready to print or save as CSV."""
    fmt = lambda mean, std: f"{round(float(mean), 3)}±{round(float(std), 3)}"
    return pd.DataFrame([{
        'Model': name,
        'Min DSC': round(float(summary['min_dsc'][0]), 3),
        'DSC': fmt(*summary['dsc']),
        'Surface DSC': fmt(*summary['surface_dsc']),
        '95% HD': fmt(*summary['hd95']),
        'RAVD %': fmt(*summary['ravd']),
    }])


def register(reg_model, dataloader, device, amp=False):
    """Warp each source label onto its target with a trained CA registration network.

    The network runs in eval mode, so labels are warped with nearest neighbour (hard labels).
    The loader yields (source, target, source_label, target_label) batches.
    amp=True runs the network under autocast (fp16 on the GPU).
    Returns (target_labels, warped_source_labels), concatenated over batches, on the CPU.
    """
    reg_model.to(device).float().eval()
    targets, warped = [], []
    with torch.no_grad():
        for source, target, source_label, target_label in dataloader:
            with autocast(enabled=amp):
                _, _, source_to_target_label, _, _ = reg_model(
                    source.to(device).float(), target.to(device).float(), source_label.to(device).float())
            warped.append(source_to_target_label.cpu())
            targets.append(target_label)
    return torch.cat(targets), torch.cat(warped)


def load_segmenter(checkpoint, in_shape, in_channels, nb_features=(4, 8, 16, 32)):
    """Load one trained downstream segmenter (e.g. <experiment>/model_params/unet_fold{i}.pt) on the CPU."""
    model = SegmentationUNet(in_shape=in_shape, infeatures=in_channels, outfeatures=2,
                             nb_features=list(nb_features), max_pool=2)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    return model


def predict(seg_model, dataloader, device='cpu', split_hemispheres=False):
    """Run the segmenter over a test loader in fp32. Returns (pred, true), concatenated over batches.

    Without split_hemispheres the loader yields (image, label) batches.
    With split_hemispheres (MRI) it yields (source, target, source_label, target_label) batches;
    the TARGET volumes are cut in half along the first spatial axis and each half is predicted
    separately, so every subject contributes two rows (left and right), in that order per batch.
    """
    seg_model.to(device).eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in dataloader:
            if not split_hemispheres:
                img, label = batch
                img = img.to(device).float()
                preds.append(seg_model(img).cpu())
                trues.append(label.float())
                continue
            _, target, _, target_label = batch
            target = target.to(device).float()
            mid = target.shape[-3] // 2
            for half in (slice(None, mid), slice(mid, None)):
                preds.append(seg_model(target[..., half, :, :]).cpu())
                trues.append(target_label[..., half, :, :].float())
    return torch.cat(preds), torch.cat(trues)
