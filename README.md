# outfit-classifier

Multi-label outfit classification project built around the notebook
`outfit-classifier.ipynb`. The notebook is the main pipeline: it prepares a
leakage-safe split, builds image loaders, trains several `timm` backbones,
evaluates default and tuned thresholds, and saves the final checkpoint used by
the demo and inference notebooks.

## Project Layout

- `outfit-classifier.ipynb` - full training and evaluation pipeline.
- `final_model_test.ipynb` - checkpoint-only evaluation and baseline comparison.
- `demo_app.py` - small local web demo for image upload and random test samples.
- `outfit_inference.py` - shared inference helpers used by the demo/test flow.
- `prepare_dataset.py` - dataset preparation utility.
- `dataset/` - prepared CSV and resized image folders.
- `checkpoints/` - saved model checkpoints.
- `results/` - split files, comparison tables, plots, and reports.

## Dataset Expected By The Notebook

The notebook expects:

- `dataset/dataset.csv`
- `dataset/train_384/`
- `dataset/train_512/`

`dataset/dataset.csv` must contain:

- `filename`
- optional `split`
- one binary column per clothing/accessory label

Current checked dataset shape:

- `dataset/dataset.csv`: 9,386 rows
- label columns: 24
- train split: 6,598 rows
- validation split: 641 rows
- test split: 641 rows

The split files are written to:

- `results/train_split.csv`
- `results/val_split.csv`
- `results/test_split.csv`
- `results/split_summary.csv`

## Dataset Preparation

`prepare_dataset.py` converts the original image-label export into the prepared
training folder consumed by the notebook.

Typical usage:

```bash
python prepare_dataset.py \
  --json checkpoint_static.json \
  --imgs train \
  --out dataset \
  --val-size 0.2 \
  --seed 42 \
  --resize-mode pad \
  --aug-max-transforms 4 \
  --clean-output
```

The script performs these steps:

1. Loads the source JSON labels.
2. Supports both dictionary-style labels and list-style labels with a `classes`
   field.
3. Removes dress-only rows by default, because those samples are weak for
   multi-label outfit classification. Use `--keep-dress-only` to keep them.
4. Optionally drops selected label columns with `--drop-labels`.
5. Finds rare labels using `--aug-rare-threshold`; default is labels present in
   fewer than `10%` of rows.
6. Augments images containing rare labels, up to `--aug-max-transforms`
   variants per source image.
7. Writes `train.csv`, `val.csv`, and `dataset.csv`.
8. Resizes every retained image into both `384x384` and `512x512` folders.

Available rare-label augmentation transforms:

- horizontal flip
- vertical flip
- 90 degree rotation
- 270 degree rotation
- brighter image
- darker image
- stronger contrast
- stronger saturation
- lower saturation
- sharper image

The resize mode defaults to padded square resize:

```text
ImageOps.pad(image, (size, size), color=(255, 255, 255))
```

This preserves the full clothing item instead of center-cropping it away. Use
`--resize-mode crop` only if you intentionally want square center crops.

Output files from preparation:

- `dataset/train.csv`
- `dataset/val.csv`
- `dataset/dataset.csv`
- `dataset/train_384/`
- `dataset/train_512/`
- `dataset/augmentation_report.txt`
- `dataset/missing_images_384.txt` when needed
- `dataset/missing_images_512.txt` when needed

The notebook later creates its own leakage-safe train/validation/test split from
`dataset/dataset.csv`. The `train.csv` and `val.csv` from preparation are useful
as preparation artifacts, but the final modeling split is the notebook split.

## Notebook Pipeline

Run `outfit-classifier.ipynb` from top to bottom after the prepared dataset is in
place.

### 1. Imports And Config

The first code cell defines `CONFIG` and `MODEL_FAMILIES`.

Important config values:

- `compare_image_size = 384`
- `final_image_size = 512`
- `batch_size_384 = 4`
- `batch_size_512 = 4`
- `grad_accum_steps_384 = 4`
- `grad_accum_steps_512 = 4`
- `epochs_512 = 20`
- `head_only_epochs_512 = 2`
- `early_stopping_patience = 6`
- `loss_name = "focal_bce"`
- `pos_weight_cap = 6.0`
- `tune_thresholds = True`
- `use_tta_eval = True`

Configured model families:

- `convnext_base_384`: `convnext_base.fb_in22k_ft_in1k_384`
- `swin_base_384`: `swin_base_patch4_window12_384.ms_in22k_ft_in1k`
- `maxvit_small_384`: `maxvit_small_tf_384.in1k`

### 2. Runtime Setup

The notebook seeds Python, NumPy, and PyTorch, chooses CUDA when available, and
prints GPU memory information. If CUDA is unavailable it falls back to CPU, but
full training is intended for GPU.

### 3. Leakage-Safe Split

The dataset already includes augmented image filenames such as
`*_aug_hflip.jpg`. Splitting rows directly would leak near-duplicates across
train/validation/test.

The notebook:

- extracts a `base_id` from each filename
- groups every augmented variant with its original image
- creates train/validation/test groups
- keeps augmented rows only in training
- keeps validation and test on original images only
- writes split CSVs to `results/`

This is important. A random row split would inflate validation/test performance.

### 4. Dataset And DataLoaders

`OutfitDataset` loads resized images from `dataset/train_384/` or
`dataset/train_512/`.

The notebook can cache decoded `uint8` images in RAM:

- `cache_images_in_ram = True`
- `num_workers = 0`

`num_workers = 0` is intentional because RAM caching with multiple workers would
duplicate the cache and waste memory.

Training augmentation currently includes:

- horizontal flip
- small rotation
- random perspective
- color jitter
- random erasing after tensor conversion

Validation and test use only conversion and normalization.

### 5. Model Definition

The model is a `timm` backbone with a small multi-label head:

- backbone created with `num_classes=0`
- dropout
- linear layer to 24 labels

The model builder accepts `image_size`. This matters for Swin and MaxViT because
those models enforce input size. ConvNeXt is more flexible, but Swin/MaxViT need
to be initialized for the requested size.

### 6. Training And Evaluation Helpers

The notebook uses:

- focal BCE loss with class `pos_weight`
- weighted sampling for imbalanced labels
- gradient accumulation
- staged training
- validation macro F1 for checkpoint selection
- optional horizontal-flip TTA for final evaluation
- optional per-class threshold tuning

The loss is `FocalBCELoss`, using `pos_weight` capped by `pos_weight_cap`.

Threshold reporting is explicit. Result tables include default-threshold and
tuned-threshold columns, for example:

- `val_macro_f1_default`
- `val_macro_f1_tuned`
- `test_macro_f1_default`
- `test_macro_f1_tuned`

Do not assume tuned thresholds are automatically better. They are selected on
validation and can overfit calibration, so the held-out test columns matter.

### 7. Train One Model

`train_one_model(...)` handles one complete training run:

- creates loaders for the selected image size
- builds the model
- optionally initializes from an earlier checkpoint
- trains the head first
- unfreezes the backbone for full fine-tuning
- saves the best checkpoint by validation macro F1
- reloads the best checkpoint
- evaluates validation/test metrics
- tunes thresholds if enabled
- stores threshold metadata in the checkpoint
- returns a row for comparison tables and per-class F1 plots

For 384-to-512 continuation, checkpoint loading skips only incompatible
size-specific tensors. This is mainly for positional buffers in size-sensitive
models.

### 8. Smoke Test

The smoke-test cell builds the first configured model and pulls one batch from
the training loader. Use this before starting a long run. If it fails, fix the
dataset/model setup before training.

### 9. Quick 384px Comparison

The notebook has a cell that compares all configured model families at 384px and
writes:

- `results/comparison_384_quick.csv`
- `results/comparison_384_quick.md`
- `results/model_metrics_384_quick.png`
- `results/training_curves_384_quick.png`
- `results/per_class_f1_384_quick.csv`
- `results/per_class_f1_384_quick.png`

The current quick comparison found `swin_base_384` as the strongest 384px
candidate in `results/comparison_384_quick.md`.

If the notebook was restarted and you already know the winner, you can skip the
quick-comparison cell and directly configure:

```python
best_base_model_name = "swin_base_384"
best_model_info = MODEL_FAMILIES[best_base_model_name]
quick_checkpoint = Path("checkpoints/swin_base_384_quick_384_best.pth")
```

Then train the longer 384px winner run from that checkpoint when available.

### 10. Winner, Frozen Baseline, And Final 512px Training

The final training section is split into stages:

- `swin_base_384_winner384`: longer 384px training for the chosen family
- `swin_base_384_frozen_baseline`: same backbone with backbone frozen and only
  the classification head trained
- `swin_base_384_final512`: final 512px continuation from the winner checkpoint

After final training, the notebook copies the final checkpoint to:

```text
checkpoints/best_model.pth
```

That file is the main inference checkpoint expected by `demo_app.py` and
`final_model_test.ipynb`.

## Outputs

Common checkpoint names:

- `checkpoints/<model>_quick_384_best.pth`
- `checkpoints/<model>_winner384_384_best.pth`
- `checkpoints/<model>_frozen_baseline_384_best.pth`
- `checkpoints/<model>_final512_512_best.pth`
- `checkpoints/best_model.pth`

Common result files:

- `results/comparison_384_quick.csv`
- `results/comparison_384_quick.md`
- `results/comparison_final.csv`
- `results/comparison_final.md`
- `results/main_model.csv`
- `results/model_metrics_384_quick.png`
- `results/model_metrics_final.png`
- `results/training_curves_384_quick.png`
- `results/training_curves_final.png`
- `results/per_class_f1_384_quick.csv`
- `results/per_class_f1_384_quick.png`
- `results/per_class_f1_final.csv`
- `results/per_class_f1_final.png`
- `results/dataset_summary.json`

## Demo

Run the local demo after `checkpoints/best_model.pth` exists:

```bash
python demo_app.py
```

The app prints the URL it selected. If port `8000` is busy, it tries the next
available port.

The demo supports:

- choosing an image
- predicting labels
- loading a random test image
- clearing the view
- changing the global threshold interactively

## Practical Notes

- The notebook is the source of truth for training.
- `final_model_test.ipynb` is for checkpoint-only evaluation, not retraining.
- `demo_app.py` uses `checkpoints/best_model.pth`.
- If the kernel restarts after quick comparison, you do not need to rerun all
  model-family checks if the quick checkpoints already exist.
- If Swin or MaxViT crashes on image size, rerun the model-definition and
  training-helper cells so the `image_size`-aware builder is active.
- Compare default and tuned test metrics before deciding which result to report.
