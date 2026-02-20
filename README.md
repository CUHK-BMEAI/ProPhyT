# ProPhyT: Prototype-based Physiological Transfer for NCCT-only Stroke Tissue-Window Segmentation

> **Prototype-based Physiological Transfer Enables NCCT-only Stroke Tissue-Window Segmentation Under Missing Perfusion**
> MICCAI 2026 (under review)

## Abstract

Accurate delineation of ischemic core and salvageable penumbra is critical for guiding reperfusion therapy in acute ischemic stroke (AIS). While CT perfusion (CTP) provides quantitative hemodynamic parameters for tissue-window assessment, its availability remains limited in many emergency settings. In contrast, non-contrast CT (NCCT) is nearly universally accessible but lacks explicit perfusion information, making core and penumbra segmentation highly challenging.

We study stroke tissue-window segmentation under missing perfusion, where only NCCT is available at inference. In this setting, specialist NCCT models struggle due to subtle contrast and limited supervision, while reconstructing CTP from NCCT constitutes an ill-posed inverse problem prone to hemodynamic hallucination.

To address this, we propose **ProPhyT**, a prototype-based physiological transfer framework that injects CTP-derived hemodynamic semantics into NCCT-based segmentation without reconstructing perfusion maps. ProPhyT consists of three processes:

1. **Hemodynamic Prototype Support Bank Construction** – A self-supervised CTP encoder learns a hemodynamic embedding space and CTP features are clustered to form representative physiological prototypes.
2. **Cross-Modal Prototype Retrieval Learning** – An NCCT encoder is aligned to the CTP embedding space to enable prototype retrieval from NCCT alone.
3. **Physiological Prototype-Conditioned Segmentation** – Retrieved prototypes are converted into similarity-aware location prompts to parameter-efficiently fine-tune a medical foundation segmentation model for core and penumbra prediction.

Extensive experiments demonstrate consistent improvements over top-performing baselines for ischemic core and penumbra segmentation, supporting more reliable reperfusion decision-making in CTP-limited clinical settings.

## Method Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         ProPhyT Pipeline                         │
├────────────────────┬─────────────────────┬───────────────────────┤
│   Stage 1a         │   Stage 1b          │   Stage 2             │
│   CTP Prototype    │   NCCT Encoder      │   Prototype-Conditioned│
│   Bank Construction│   Alignment         │   Segmentation        │
│                    │                     │                       │
│  Self-supervised   │  Align NCCT encoder │  Prototype retrieval  │
│  CTP encoder +     │  to CTP embedding   │  → similarity-aware   │
│  K-means clustering│  space via          │  prompts → SAM-based  │
│  → Prototype bank  │  contrastive loss   │  segmentation         │
└────────────────────┴─────────────────────┴───────────────────────┘
```

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.13
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [SparK](https://github.com/keyu-tian/SparK) (included under `SparK/`)

Install dependencies:

```bash
pip install torch torchvision
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Data Preparation

Organize your dataset directory as follows:

```
data_penumbra_noblank_withvalid/
├── train/
│   ├── ncct/
│   ├── ctp/
│   └── masks/
├── val/
└── test/
```

## Usage

### Stage 1a – Hemodynamic Prototype Bank Construction

Train the self-supervised CTP encoder and build the prototype bank. Refer to the SparK pre-training scripts under `SparK/pretrain/`.

### Stage 1b – Cross-Modal Prototype Retrieval Learning

Train the NCCT encoder to retrieve CTP prototypes:

```bash
bash run_train_baseline_sam.sh
```

### Stage 2 – Prototype-Conditioned Segmentation

Fine-tune the SAM-based segmentation model conditioned on physiological prototypes:

```bash
bash run_train_ctp_prompt.sh
```

Key arguments:

| Argument | Description | Default |
|---|---|---|
| `--stage1b_ckpt` | Path to Stage 1b checkpoint | — |
| `--prototype_bank` | Path to `.npy` prototype bank | — |
| `--fusion_mode` | Prototype fusion mode (`dense_prompt`) | `dense_prompt` |
| `--latent_dim` | Embedding dimension | `256` |
| `--top_k` | Number of prototypes to retrieve | `5` |
| `--temperature` | Softmax temperature for retrieval | `0.1` |

### Inference

```bash
bash run_test_ctp_prompt.sh
```

## Evaluation

```bash
bash run_test_ctp_prompt.sh   # ProPhyT
bash run_test_baseline_sam.sh # Baseline SAM
```

Metrics (Dice, IoU, etc.) are computed via `metrics.py`.

## Repository Structure

```
ProPhyT/
├── SparK/                   # Self-supervised CTP encoder (SparK backbone)
├── segment_anything/        # SAM model
├── cpal_sam_modules.py      # ProPhyT modules (FPN, PrototypeRetriever, etc.)
├── DataLoader.py            # Dataset utilities
├── train.py                 # Baseline SAM training
├── train_cpal_sam_v2.py     # ProPhyT training (Stage 2)
├── test.py                  # Baseline SAM testing
├── test_cpal_sam_v2.py      # ProPhyT testing
├── metrics.py               # Evaluation metrics
├── utils.py                 # General utilities
├── run_train_ctp_prompt.sh  # Train ProPhyT
├── run_test_ctp_prompt.sh   # Test ProPhyT
├── run_train_baseline_sam.sh
└── run_test_baseline_sam.sh
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{prophyt2026,
  title     = {Prototype-based Physiological Transfer Enables NCCT-only Stroke Tissue-Window Segmentation Under Missing Perfusion},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026},
  note      = {Code available upon acceptance}
}
```

## License

Code will be released upon paper acceptance.
