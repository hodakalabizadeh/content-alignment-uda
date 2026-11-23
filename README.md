# Content Alignment for unsupervised domain adaptation

![Framework overview](figures/framework.png)

Code for **"Unsupervised Domain Adaptation via Content Alignment for Hippocampus Segmentation"**,
Hoda Kalabizadeh, Ludovica Griffanti, Pak-Hei Yeung, Ana I. L. Namburete, Nicola K. Dinsdale and
Konstantinos Kamnitsas, ICCV Workshops 2025 ([arXiv:2510.13075](https://arxiv.org/abs/2510.13075),
[doi:10.1109/ICCVW69036.2025.00119](https://doi.org/10.1109/ICCVW69036.2025.00119)).

Content Alignment (CA) adapts a segmenter from a labelled source domain to an unlabelled target
domain by addressing two kinds of shift separately:

- **Style** (intensity) shift is removed by preprocessing: histogram standardisation, z-normalisation
  and a crop to the region of interest.
- **Content** (shape) shift is removed by registration. A bidirectional diffeomorphic registration
  network **R** warps each source image and its label towards a target image. It is trained jointly with:
  - a discriminator **D**, which pushes warped source images to look like target images;
  - a segmenter **S**, trained on the source only, which pushes the warped images to stay segmentable.

The warped source images and labels then train a downstream segmenter, which is evaluated on the
target domain. Target labels are never used for training or for model selection.

## Repository layout

```
ca_core/
  datasets/       mri.py (3D brain MRI pairs), mnist.py (Morpho-MNIST pairs),
                  MNI152_template.nii.gz (the MNI template the MRI volumes are registered to; own licence)
  models/         registration.py (R), discriminator.py (D), segmentation.py (S and downstream U-Net), warping.py
  training/       content_alignment.py (one CA epoch), downstream_segmentation.py, losses.py
  evaluation.py   Dice, surface Dice, 95% Hausdorff distance, RAVD
configs/          obhc.yaml, harp.yaml (DRYAD -> OBHC / HarP), mnist.yaml (thick -> thin digits)
scripts/          train_content_alignment.py, eval_content_alignment.py,
                  train_downstream_seg.py, eval_downstream_seg.py, slurm/ (job scripts)
requirements.txt  Python dependencies
```

## Installation

Clone the repository, then from its root:

```bash
conda create -n ca python=3.11 && conda activate ca
pip install -r requirements.txt
```

Tested with PyTorch 2.0.1 on an NVIDIA A10 GPU.

## Data

The MRI experiments use DRYAD as the source domain and OBHC or HarP as the target domain. Expected input:

- NIfTI volumes, already affinely registered to 1 mm MNI space, skull-stripped and bias-corrected (N4).
  The MNI template used is `ca_core/datasets/MNI152_template.nii.gz` (ICBM 2009a Nonlinear Symmetric,
  197 × 233 × 189; see `ca_core/datasets/MNI152_template_NOTICE.md`); the ROI crop assumes this grid;
- a hippocampus label map for each volume (any value > 0 is foreground).

By default the configs read the data from `data/` in the repository root (`data_root`); put or link the
datasets there, or set `data_root` to your data folder. The train/test image and label folders are listed
under `data:`, relative to `data_root`. Each volume is then:

1. histogram-standardised over the brain (voxels above 10% of the volume's 99th percentile), with
   landmarks fitted on the source and target training volumes together;
2. z-normalised over the brain;
3. cropped to a (96, 128, 80) hippocampal region of interest.


## Running

From the repository root, with `PYTHONPATH` set to it:

```bash
python scripts/train_content_alignment.py --config configs/obhc.yaml --run-name ca   # train R + D + S, every CA fold
python scripts/eval_content_alignment.py  --config configs/obhc.yaml --run-name ca   # label propagation on the training data
python scripts/train_downstream_seg.py    --config configs/obhc.yaml --run-name ca   # segmenter on CA-aligned source data
python scripts/eval_downstream_seg.py     --config configs/obhc.yaml --run-name ca   # Dice etc. on the target test set
```

On a Slurm cluster, `bash scripts/slurm/run_all.sh obhc ca` submits the stages as chained jobs, with one
job per CA fold; `bash scripts/slurm/run_all.sh obhc ca_s43 43 "42 43 44"` uses CA seed 43 and three
downstream seeds. Outputs go to `outputs/<config>/<run-name>/`.

- **Ablations:** `--lambda-disc 0` and/or `--lambda-seg 0` switch off D and/or S.
- **CA training:** k-fold cross-validation (`ca.folds`, 3), split with `--seed`.
  - Each fold's CA model is trained on the other folds and early-stopped on R's validation loss on its
    own fold, which does not use target labels. `--fold k` trains a single fold.
  - Saves `ca_fold<k>.pt` and `ca_fold<k>.json` (best validation loss and epoch) per fold.
- **Choosing the CA model:** the downstream stage uses the fold with the lowest validation loss
  (`ca_core.utils.select_ca_fold`). Target labels are never used for this choice; the registration
  Dice in `ca_eval.csv` is for reporting only.
- **Downstream segmenter:** trained with k-fold cross-validation (`downstream.folds`, 3), and
  evaluated as the mean over the k fold models. Outputs of each downstream seed go to
  `downstream_seed<seed>/`.


## Citation

```bibtex
@inproceedings{kalabizadeh2025content,
  title     = {Unsupervised Domain Adaptation via Content Alignment for Hippocampus Segmentation},
  author    = {Kalabizadeh, Hoda and Griffanti, Ludovica and Yeung, Pak-Hei and Namburete, Ana I. L. and
               Dinsdale, Nicola K. and Kamnitsas, Konstantinos},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  year      = {2025},
  doi       = {10.1109/ICCVW69036.2025.00119}
}
```

## License

MIT; see [LICENSE](LICENSE). The MNI template in `ca_core/datasets/` is distributed under its own notice
(McConnell Brain Imaging Centre); see [MNI152_template_NOTICE.md](ca_core/datasets/MNI152_template_NOTICE.md).
