"""One epoch of Content Alignment (CA) training and validation.

CA trains three networks jointly on (source, target) pairs:
    R  registration network (BidirDiffeoRegistration): warps source -> target and target -> source
    D  discriminator: real target images vs source images warped to the target
    S  segmenter: trained on the labelled SOURCE domain only

Each batch:
    1. D step on its own loss:  BCE(D(target), 1) + BCE(D(R(source)), 0), averaged.
    2. S step on its own loss:  seg_loss(S(source), source_label).
    3. R step on
           lambda_sim    * [MSE(target, source->target) + MSE(source, target->source)]
         + lambda_smooth * [Smoothness(forward field) + Smoothness(inverse field)]
         + lambda_seg    * seg_loss(S(target->source), source_label)
         + lambda_disc   * BCE(D(source->target), 1)

Target labels are never used for training: they are read only to report validation Dice.
Mixed precision follows the GradScaler that is passed in: GradScaler(enabled=False) trains in fp32.
The scaler is updated after every optimiser step (up to three updates per batch), as in the original code.
"""
import numpy as np
import surface_distance as s_d
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from ..models.discriminator import Discriminator, initialize_weights
from ..models.registration import BidirDiffeoRegistration
from ..models.segmentation import SegmentationUNet
from .losses import SmoothnessLoss

REG_FEATURES = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]  # registration U-Net [encoder, decoder]
SEG_FEATURES = [4, 8, 16, 32]                                     # segmentation U-Net, per level
DISC_FEATURES = 64                                                # discriminator base width


def build_registration(shape, channels, int_steps=7):
    """The registration network R for images of `shape` (2D or 3D) with `channels` channels."""
    return BidirDiffeoRegistration(inshape=shape, in_channels=channels, nb_unet_features=REG_FEATURES,
                                   int_steps=int_steps)


def build_ca_models(shape, channels, int_steps=7):
    """R, D and S for Content Alignment training (built in this order: weight init draws from torch's RNG)."""
    reg = build_registration(shape, channels, int_steps)
    disc = Discriminator(channels, DISC_FEATURES, ndims=len(shape))
    initialize_weights(disc)
    seg = SegmentationUNet(in_shape=shape, infeatures=channels, outfeatures=2, nb_features=SEG_FEATURES, max_pool=2)
    return reg, disc, seg


def _to_device(device, *tensors):
    return [t.to(device).float() for t in tensors]


def _disc_loss(disc_model, disc_loss_func, target, source_to_target_img):
    """D's own loss: real target images -> 1, warped source images -> 0."""
    disc_real = disc_model(target).reshape(-1)
    disc_fake = disc_model(source_to_target_img.detach()).reshape(-1)
    loss_real = disc_loss_func(disc_real, torch.ones_like(disc_real))
    loss_fake = disc_loss_func(disc_fake, torch.zeros_like(disc_fake))
    return (loss_real + loss_fake) / 2


def _registration_image_loss(source, target, source_to_target_img, target_to_source_img,
                             pos_field, neg_field, lam_sim, lam_smooth):
    """Similarity (both directions) + smoothness (both fields) terms of R's loss."""
    mse, smooth = nn.MSELoss(), SmoothnessLoss()
    return (mse(target, source_to_target_img) * lam_sim
            + mse(source, target_to_source_img) * lam_sim
            + smooth(pos_field) * lam_smooth
            + smooth(neg_field) * lam_smooth)


def _step(scaler, optim):
    """Optimiser step through the GradScaler, then update the loss scale (after every step, not once per batch)."""
    scaler.step(optim)
    scaler.update()


def train_ca_epoch(reg_model, disc_model, seg_model, dataloader, reg_optim, disc_optim, seg_optim,
                   disc_loss_func, seg_loss_func, loss_weights, scaler, device):
    """Train R, D and S for one epoch. Returns the mean R loss.

    loss_weights: (lam_sim, lam_smooth, lam_disc, lam_seg). lam_disc = 0 or lam_seg = 0 switches
    that network off (it is neither trained nor used in R's loss), giving the paper's ablations.
    """
    for m in (reg_model, disc_model, seg_model):
        m.to(device).train()
    amp = scaler.is_enabled()
    lam_sim, lam_smooth, lam_disc, lam_seg = loss_weights

    total_reg_loss, batches = 0.0, 0
    for source, target, source_label, _ in tqdm(dataloader, desc="Training", leave=True):  # target label unused
        batches += 1
        source, target, source_label = _to_device(device, source, target, source_label)

        with autocast(enabled=amp):
            source_to_target_img, pos_field, _, target_to_source_img, neg_field = \
                reg_model(source, target, source_label)

        # --- D step ---
        if lam_disc > 0:
            disc_optim.zero_grad()
            loss_disc = _disc_loss(disc_model, disc_loss_func, target, source_to_target_img)
            scaler.scale(loss_disc).backward()
            _step(scaler, disc_optim)

        # --- S step (source domain only) ---
        if lam_seg > 0:
            seg_optim.zero_grad()
            with autocast(enabled=amp):
                loss_seg = seg_loss_func(seg_model(source), source_label)
            scaler.scale(loss_seg).backward()
            _step(scaler, seg_optim)

        # --- R step ---
        reg_optim.zero_grad()
        with autocast(enabled=amp):
            reg_loss = _registration_image_loss(source, target, source_to_target_img, target_to_source_img,
                                                pos_field, neg_field, lam_sim, lam_smooth)
            if lam_seg > 0:
                # the target warped into source space should be segmentable like the source
                reg_loss += seg_loss_func(seg_model(target_to_source_img), source_label) * lam_seg
        if lam_disc > 0:
            # the source warped into target space should look like a target image to D
            disc_out = disc_model(source_to_target_img).reshape(-1)
            reg_loss += disc_loss_func(disc_out, torch.ones_like(disc_out)) * lam_disc
        scaler.scale(reg_loss).backward()
        _step(scaler, reg_optim)

        total_reg_loss += reg_loss.item()

    av_loss = total_reg_loss / batches
    print(f'\nTraining set: Average loss: {av_loss:.4f}', flush=True)
    return av_loss


def validate_ca(reg_model, disc_model, seg_model, dataloader, disc_loss_func, seg_loss_func, loss_weights,
                device, amp=True):
    """Validate R, D and S for one epoch.

    Returns a dict of means over the validation set:
        'img_loss'    similarity + smoothness terms of R's loss
        'disc_loss'   D's own loss
        'seg_loss'    S's own loss on the source
        'total_loss'  R's full loss (used for early stopping)
        'dice'        warped source label vs target label (reported only)
    """
    for m in (reg_model, disc_model, seg_model):
        m.to(device).eval()  # eval mode: R warps labels with nearest neighbour
    lam_sim, lam_smooth, lam_disc, lam_seg = loss_weights

    totals = dict(img_loss=0.0, disc_loss=0.0, seg_loss=0.0, total_loss=0.0, dice=0.0)
    batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=True):
            batches += 1
            source, target, source_label, target_label = _to_device(device, *batch)

            with autocast(enabled=amp):
                source_to_target_img, pos_field, source_to_target_label, target_to_source_img, neg_field = \
                    reg_model(source, target, source_label)

            # Dice of the warped source label against the target label (reported, not optimised)
            totals['dice'] += np.mean([
                s_d.compute_dice_coefficient(target_label[i, 1].cpu().numpy().astype(bool),
                                             source_to_target_label[i, 1].cpu().numpy().astype(bool))
                for i in range(source_to_target_label.shape[0])
            ])

            if lam_disc > 0:
                totals['disc_loss'] += _disc_loss(disc_model, disc_loss_func, target, source_to_target_img).item()

            if lam_seg > 0:
                with autocast(enabled=amp):
                    totals['seg_loss'] += seg_loss_func(seg_model(source), source_label).item()

            with autocast(enabled=amp):
                reg_loss = _registration_image_loss(source, target, source_to_target_img, target_to_source_img,
                                                    pos_field, neg_field, lam_sim, lam_smooth)
                totals['img_loss'] += reg_loss.item()
                if lam_seg > 0:
                    reg_loss += seg_loss_func(seg_model(target_to_source_img), source_label) * lam_seg
            if lam_disc > 0:
                disc_out = disc_model(source_to_target_img).reshape(-1)
                reg_loss += disc_loss_func(disc_out, torch.ones_like(disc_out)) * lam_disc
            totals['total_loss'] += reg_loss.item()

    metrics = {k: v / batches for k, v in totals.items()}
    print(f"\nValidation set: Average Img loss: {metrics['img_loss']:.4f}")
    print(f"Validation set: Average Disc loss: {metrics['disc_loss']:.4f}")
    print(f"Validation set: Average Seg loss: {metrics['seg_loss']:.4f}")
    print(f"Validation set: Average Total loss: {metrics['total_loss']:.4f}")
    print(f"Validation set: Average Dice: {metrics['dice']:.4f}")

    return metrics
