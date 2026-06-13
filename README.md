# ProPhyT

[MICCAI 2026] Prototype-based Physiological Transfer Enables NCCT-only Stroke Tissue-Window
Segmentation Under Missing Perfusion.

## Overview

ProPhyT targets ischemic core and penumbra segmentation when only non-contrast CT
(NCCT) is available at inference time. The method transfers CTP-derived
hemodynamic semantics to NCCT segmentation through a prototype support bank,
without reconstructing perfusion maps.

The repository contains:

- `train_cpal_sam_v2.py` and `test_cpal_sam_v2.py`: ProPhyT / CM-CPAL-SAM
  training and testing entry points.
- `train.py` and `test.py`: baseline SAM-Med2D fine-tuning and testing.
- `cpal_sam_modules.py`: prototype retrieval and prompt-conditioning modules.
- `DataLoader.py`, `utils.py`, `metrics.py`: data loading, losses, transforms,
  logging, mask saving, and segmentation metrics.
- `segment_anything/`: SAM-Med2D model code used by the training scripts.
- `SparK/`: encoder components used by the physiological prototype modules.
- `scripts/`: data split and cross-validation result utilities.
- `run_*.sh`: reproducible shell wrappers for common training and testing jobs.

## Environment

Create a Python environment with PyTorch, then install the remaining Python
dependencies:

```bash
pip install -r requirements.txt
```

Install the CUDA-enabled PyTorch build that matches your driver and CUDA runtime
before running GPU training.

## Expected Data Layout

The loaders expect a dataset root with images, masks, and JSON mappings:

```text
data_penumbra_noblank_withvalid/
  images/
    <subject>_<slice>.png
  masks/
    <subject>_<slice>_core_000.png
    <subject>_<slice>_penumbra_000.png
  image2label_train.json
  label2image_valid.json
  label2image_test.json
```

`image2label_*.json` maps each image path to its two mask paths. `label2image_*.json`
maps each mask path back to the corresponding image path.

For subject-level cross-validation, generate fold JSONs with:

```bash
python scripts/create_subject_5fold_splits.py \
  --data-root data_penumbra_noblank_withvalid \
  --output-root data_penumbra_noblank_withvalid_5fold_subject_seed42 \
  --n-folds 5 \
  --seed 42
```

## Checkpoints

The ProPhyT wrappers use these default checkpoint paths:

```text
sam-med2d_b1106.pth
scripts/logs/cpal_stage1b_v2/best_model.pth
scripts/logs/cpal_stage1a_post/ctp_prototypes_post.npy
```

You can either place files at those paths or override them with environment
variables in the shell wrappers, for example:

```bash
STAGE1B_CKPT=/path/to/best_model.pth \
PROTOTYPE_BANK=/path/to/ctp_prototypes_post.npy \
bash run_train_ctp_prompt.sh
```

## Quick Start

Train ProPhyT with dense prototype prompts:

```bash
bash run_train_ctp_prompt.sh
```

Test ProPhyT:

```bash
bash run_test_ctp_prompt.sh
```

Run 5-fold subject-level ProPhyT training and testing:

```bash
bash run_train_test_ctp_prompt_5fold.sh all
```

Run the SAM-Med2D baseline:

```bash
bash run_train_baseline_sam.sh
bash run_test_baseline_sam.sh
```

Most wrappers expose configuration through environment variables, including
`PYTHON_BIN`, `WORK_DIR`, `DATA_PATH`, `DEVICE`, `BATCH_SIZE`, `EPOCHS`,
`STAGE1B_CKPT`, and `PROTOTYPE_BANK`.

## Outputs

Training outputs are written under:

```text
workdir/
  logs/
  models/<run_name>/
```

Testing writes summary and per-slice CSV files to `workdir/` or
`workdir/cv_results/`, depending on the wrapper.

## Citation

If you use this work, please cite the paper once publication metadata are
available:

```bibtex
@article{prophyt2025,
  title   = {Prototype-based Physiological Transfer Enables NCCT-only Stroke Tissue-Window Segmentation Under Missing Perfusion},
  author  = {Ying Dan*, Zhe Xu*, Longxi Zhou, Lu Zhang, Xiangyuan Ma, Raymond Tong, Yading Yuan},
  journal = {MICCAI},
  year    = {2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
