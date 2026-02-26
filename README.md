# ST-AlignNet: Semantic Alignment with Temporal Saliency Filtering for Emotion Recognition

> **Preprint | Submitted to Elsevier | February 2026**  
> Ruimin Zhang · Columbia University, Department of Statistics  
> 📧 rz2737@columbia.edu

---

## 📣 News

- **2026-02-09**: Preprint submitted. Code is publicly available. Paper link will be updated upon acceptance.

---

## Introduction

Fusing heterogeneous modalities such as **EEG** and **facial expressions** remains a significant challenge in emotion recognition due to inherent **semantic misalignment** and **temporal asynchrony**. ST-AlignNet addresses these two core problems via:

- **Bidirectional Semantic Alignment (BSA)**: A symmetric cross-attention mechanism that dynamically aligns heterogeneous EEG and facial features in a shared latent space, enabling the model to adaptively trust the more reliable modality (e.g., deferring to EEG in poker face scenarios where facial expressions are ambiguous).
- **Saliency-Aware Temporal Aggregation (SATA)**: A learnable `[CLS]`-token-based attention pooling that identifies and up-weights emotionally significant frames, suppressing irrelevant neutral segments that would otherwise dilute salient emotional cues under standard mean pooling.

<p align="center">
  <img src="assets/framework.png" width="900" alt="ST-AlignNet Architecture"/>
</p>

> The overall framework consists of three stages: (1) Feature Encoding for EEG (HCNN) and facial inputs (ResNet-50); (2) Bidirectional Semantic Alignment (BSA) via symmetric cross-attention; and (3) Saliency-Aware Temporal Aggregation (SATA) to capture sparse emotional bursts for classification.

---

## Results

### DEAP Dataset (Accuracy %)

| Method | Dep. Valence | Dep. Arousal | Indep. Valence | Indep. Arousal |
|---|---|---|---|---|
| DGCNN | 66.38 ± 5.26 | 68.62 ± 8.15 | 57.07 ± 7.34 | 63.09 ± 9.38 |
| TSception | 74.38 ± 8.18 | 76.21 ± 8.84 | — | — |
| M2S | 75.03 ± 10.82 | 77.24 ± 12.07 | — | — |
| 3D-CNN | 89.45 ± 4.51 | 90.42 ± 3.72 | 56.25 ± 1.86 | 56.77 ± 7.29 |
| EEGNet | 90.20 ± 2.43 | 91.22 ± 2.60 | 61.70 | 59.00 |
| UAGCFNet | 88.71 ± 4.40 | 87.13 ± 5.76 | 69.62 ± 7.32 | 70.62 ± 7.96 |
| CADD-DCCNN | 90.97 ± 13.96 | 92.42 ± 12.72 | 69.45 ± 5.60 | 70.50 ± 9.39 |
| MAS-DGAT-Net | 94.69 ± 4.11 | 95.21 ± 4.52 | 64.76 ± 5.02 | 62.86 ± 5.34 |
| ESC-GAN | 96.33 | 96.68 | — | — |
| SCCapsNet | 96.45 ± 3.99 | 96.84 ± 4.16 | 58.33 ± 7.70 | 62.13 ± 10.64 |
| MDNet | — | — | 65.30 | 61.50 |
| **Ours (ST-AlignNet)** | **97.95 ± 2.04** | **97.90 ± 3.38** | **70.86 ± 2.32** | **70.73 ± 1.35** |

ST-AlignNet achieves **state-of-the-art** on both settings. In the subject-independent setting, our variance (±2.32) is **3× lower** than UAGCFNet (±7.32), demonstrating superior robustness to inter-subject variability.

---

## Method

### 1. Unimodal Feature Encoding

- **EEG Encoder (HCNN)**: Hierarchical spatial + temporal convolutions over the spatio-temporal EEG tensor `(C × T)`, followed by learnable positional embeddings → `Z_eeg ∈ ℝ^{T×128}`.
- **Facial Encoder**: ResNet-50 (pretrained on ImageNet) extracts 2048-dim per-frame features from MTCNN-aligned face crops (`X_face ∈ ℝ^{128×2048}`). A Conv1D projection maps these to `d=128`, and adaptive temporal pooling synchronizes the frame sequence to EEG resolution → `Z_face ∈ ℝ^{T×128}`.

### 2. Bidirectional Semantic Alignment (BSA)

Symmetric multi-head cross-attention between modalities (`our_method/fusion.py → CrossModalFusion`):

```python
eeg2,  _ = MultiheadAttention(Q=eeg,    K=facial, V=facial)
face2, _ = MultiheadAttention(Q=facial, K=eeg,    V=eeg)
Z̃_eeg  = LayerNorm(eeg   + eeg2)
Z̃_face = LayerNorm(facial + face2)
Z_fused = (Z̃_eeg + Z̃_face) / 2      # λ = 0.5, optimal
```

### 3. Saliency-Aware Temporal Aggregation (SATA)

A learnable `[CLS]` token is prepended to `Z_fused` and processed by global self-attention (`our_method/fusion.py → AttentionPooling`):

```python
cls_token = nn.Parameter(torch.randn(1, 1, 128))
x   = cat([cls_token.repeat(B, 1, 1), Z_fused], dim=1)  # [B, T+1, 128]
out = MultiheadAttention(x, x, x)
z_cls = out[:, 0, :]    # saliency-weighted representation → classifier
```

Neutral frames receive near-zero attention weights (`α_t ≈ 0`), effectively acting as an intelligent key-frame selector.

### 4. Optimization

- **Subject-dependent**: Adam, weighted cross-entropy (class-frequency inverse weights), `label_smoothing=0.0`
- **Subject-independent**: AdamW (`weight_decay=1e-4`), warmup (5 epochs) + cosine annealing, `label_smoothing=0.05`, gradient clipping (`max_norm=1.0`), early stopping (patience=8)

---

## Ablation Study

| Variant | BSA | SATA | Dep. Valence | Dep. Arousal | Indep. Valence | Indep. Arousal |
|---|:---:|:---:|---|---|---|---|
| Concat & Mean | ✗ | ✗ | 90.20 ± 2.55 | 90.50 ± 3.80 | 59.50 ± 2.45 | 59.85 ± 1.80 |
| w/o BSA | ✗ | ✓ | 96.15 ± 2.10 | 95.80 ± 3.50 | 68.20 ± 2.10 | 68.45 ± 1.65 |
| w/o SATA | ✓ | ✗ | 97.10 ± 2.25 | 96.95 ± 3.45 | 69.55 ± 2.35 | 69.80 ± 1.50 |
| **Full Model** | ✓ | ✓ | **97.95 ± 2.04** | **97.90 ± 3.38** | **70.86 ± 2.32** | **70.73 ± 1.35** |

BSA and SATA are **synergistic**: BSA resolves semantic misalignment across modalities, while SATA selects emotionally salient moments in time. Together they yield a **>4% gain** over the concatenation baseline.

---

## Installation

```bash
git clone https://github.com/Elysia210/ST-AlignNet-Semantic-Alignment-with-Temporal-Saliency-Filtering-for-Emotion-Recognition.git
cd ST-AlignNet-Semantic-Alignment-with-Temporal-Saliency-Filtering-for-Emotion-Recognition
pip install torch torchvision numpy scipy scikit-learn pandas tensorboard facenet-pytorch
```

**Core requirements**: Python 3.8+, PyTorch 1.12+, CUDA 11.3+

---

## Datasets

| Dataset | Subjects | Stimuli | EEG | Link |
|---|---|---|---|---|
| DEAP | 32 | 40 music videos (60s each) | 32-ch @ 128Hz | [Download](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) |
| MAHNOB-HCI | 27 | 20 emotion clips | 32-ch | [Download](https://mahnob-db.eu/hci-tagging/) |

Both datasets provide continuous affective ratings for Valence and Arousal on a **1–9 scale**, binarized at threshold = 5 for classification. Both datasets require signing an EULA with the dataset providers. After obtaining access, place data as follows:

```
data/
├── DEAP/
│   ├── data_preprocessed_python/    # raw EEG .dat files
│   ├── face_video/                  # facial video files
│   ├── aligned_dependent_data/      # generated by Step 3 below
│   └── aligned_independent_data/    # generated by Step 3 below
└── MAHNOB-HCI/
    └── Sessions/
```

Labels are binarized at threshold = 5 (< 5 → Low, ≥ 5 → High) for Valence and Arousal independently.

---

## Preprocessing

**Step 1: Extract facial video frames**
```bash
python extract_video_frames.py
```

**Step 2: Extract ResNet-50 facial features**
```bash
python our_method/resnet_npy_fast.py
python our_method/resnet_png_to_npy.py
```

**Step 3: Preprocess and align EEG + facial features**
```bash
# DEAP subject-dependent
python data_preprocessing.py

# DEAP subject-independent
python data_preprocessing_2.py

# MAHNOB-HCI
python data_preprocessing_mahnob.py
```

**Step 4: Build pickle files for training**
```bash
python our_method/make_pkl_ind_python.py           # LOSO splits
python our_method/preprocess_dependent_pkl.py      # 10-fold splits
python our_method/preprocess_independent_pkl.py
```

EEG signals are bandpass filtered (4–45 Hz), Z-score normalized per channel, and downsampled to T=128 steps. Facial features are adaptively pooled to T=128 to match EEG resolution.

---

## Training

### Subject-Dependent (10-fold cross-validation per subject)

```bash
cd our_method
python main.py \
    --epochs 60 \
    --batch_size 64 \
    --lr 1e-4 \
    --label_smoothing 0.0 \
    --data_path /path/to/DEAP/aligned_dependent_data \
    --save_dir ./results/deap_dep \
    --target_dim 0      # 0: Valence  |  1: Arousal
```

For chain (warm-start across folds) training (run from **repo root**, not `our_method/`):
```bash
python train_dependent.py \
    --epochs 60 --batch_size 64 --lr 1e-4 \
    --data_path /path/to/DEAP/make_data \
    --save_dir ./output/binary_dep \
    --freeze_epochs 5         # freeze encoders for first 5 epochs
```
> `--init_ckpt /path/to/pretrained.pt` can optionally load a warm-start checkpoint.

### Subject-Independent (Leave-One-Subject-Out)

```bash
cd our_method
python main_loso.py \
    --epochs 60 \
    --batch_size 64 \
    --lr 1e-4 \
    --label_smoothing 0.05 \
    --data_path /path/to/DEAP/aligned_independent_data \
    --save_dir ./results/deap_si \
    --target_dim 0             # 0: Valence  |  1: Arousal
    --binary_threshold 5.0     # binarization threshold
```
> **Note**: `--folds` defaults to `range(22)`. For DEAP (32 subjects), pass `--folds $(seq 0 31)`.

### Baseline Single-Modality & Fusion Experiments

```bash
# EEG-only (choose encoder: EEGNet | DGCNN | LGGNet | TSception | GCBNet)
python train_final.py \
    --dataset deap --task classification --eval_mode subject_independent \
    --data_path data/DEAP/processed --modality eeg --encoder eegnet \
    --batch_size 64 --epochs 80 --lr 1e-4 \
    --output_path results/exp1/deap_si_cls/eeg_eegnet

# Facial-only (choose encoder: C3D | SlowFast | VideoSwin | LogoFormer | EST)
python train_final.py \
    --dataset deap --task classification --eval_mode subject_independent \
    --data_path data/DEAP/processed --modality facial --encoder est \
    --output_path results/exp1/deap_si_cls/facial_est

# Run all baselines at once
bash scripts/run_exp1_deap_si.sh
```

### Analyze & Summarize Results

```bash
python scripts/analyze_results.py \
    --results_dir results/exp1/deap_si_cls \
    --output_file results/exp1/deap_si_cls_summary.csv
```

---

## Key Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| Embedding dim `d` | 128 | Shared latent space for EEG + facial |
| Attention heads (BSA + SATA) | 4 | `CrossModalFusion` + `AttentionPooling` |
| BSA layers `L` | 3 | Optimal; L=1→60.1%, L=3→70.86%, L=4→68.2% (overfits) |
| Fusion weight `λ` | 0.5 | Peak at λ=0.5; facial-only(λ=0): 69.69%, EEG-only(λ=1): 66.05% |
| Label smoothing ε | 0.05 | Peak at ε=0.05; ε=0→69.50%, ε>0.1 degrades |
| Weight initialization | Xavier | Applied to all trainable parameters |
| Optimizer | Adam / AdamW | Dep.: Adam; Indep.: AdamW (`weight_decay=1e-4`) |
| LR scheduler | — / Warmup+Cosine | Dep.: none; Indep.: 5ep warmup + cosine decay to 1×10⁻⁶ (paper §4.2 describes cosine over 100ep) |
| Learning rate | 1×10⁻⁴ | Both settings |
| Min learning rate | 1×10⁻⁶ | Cosine annealing lower bound (paper §4.2) |
| Batch size | 64 | 2× NVIDIA V100 |
| Epochs | 60–80 | Early stopping patience=8 (LOSO) |
| Gradient clipping | 1.0 | Subject-independent only |
| Classifier head | Linear(128→64)→ReLU→Dropout(0.5)→Linear(64→2) | |
| Eval metrics | Accuracy + F1-Score | Both reported per paper §4.4 |

---

## Repository Structure

```
├── our_method/                          # Core ST-AlignNet implementation
│   ├── encoder.py                       # EEGEncoder + FacialEncoder
│   ├── fusion.py                        # CrossModalFusion (BSA) + AttentionPooling (SATA)
│   ├── main.py                          # Subject-dependent training (10-fold)
│   ├── main_dep.py                      # Chain warm-start dependent training
│   ├── main_loso.py                     # Subject-independent LOSO training
│   ├── main_subject_independent.py      # Subject-independent (alternative entry)
│   ├── evaluate.py                      # Metrics, calibration (Platt/Temp scaling)
│   ├── Multimodal_dataset.py            # Dataset & DataLoader builders
│   ├── make_pkl_ind_python.py           # Build LOSO pickle files
│   ├── preprocess_dependent_pkl.py      # Build 10-fold pickle files
│   ├── preprocess_independent_pkl.py    # Build independent pickle files
│   ├── extract_video_frames_all.py      # Extract frames for all videos
│   ├── resnet_npy_fast.py               # ResNet-50 feature extraction
│   ├── resnet_png_to_npy.py             # Frame-to-npy conversion
│   └── repair_missing.py               # Repair missing feature files
├── models/                              # Baseline model implementations
│   └── ...                              # EEGNet, DGCNN, LGGNet, C3D, SlowFast, etc.
├── scripts/
│   ├── eeg_only.py                      # EEG-only training wrapper
│   ├── facial_only.py                   # Facial-only training wrapper
│   ├── multimodal.py                    # Multimodal fusion training wrapper
│   ├── analyze_results.py               # Results aggregation & CSV export
│   └── run_exp1_deap_si.sh              # Full baseline experiment runner
├── data_preprocessing.py                # DEAP subject-dependent preprocessing
├── data_preprocessing_2.py             # DEAP subject-independent preprocessing
├── data_preprocessing_mahnob.py        # MAHNOB-HCI preprocessing
├── extract_video_frames.py             # Facial frame extraction
├── fusion_modules.py                   # Standalone BSA & SATA module exports
├── train_dependent.py                  # Top-level dependent training wrapper
├── train_independent.py                # Top-level independent training wrapper
├── train_fusion.py                     # Top-level multimodal fusion training
├── run_deap_eeg.sh                     # Quick run: EEG-only on DEAP
├── run_deap_facial.sh                  # Quick run: Facial-only on DEAP
└── assets/
    └── framework.png                   # Architecture figure (Fig. 1 from paper)
```

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{zhang2026stalignnet,
  title   = {ST-AlignNet: Semantic Alignment with Temporal Saliency Filtering for Emotion Recognition},
  author  = {Zhang, Ruimin},
  journal = {Preprint submitted to Elsevier},
  year    = {2026}
}
```


---

## Acknowledgements

The EEG encoder is inspired by [EEGNet](https://github.com/vlawhern/arl-eegmodels) and [DGCNN](https://github.com/xueyunlong12589/DGCNN). The facial encoder uses [ResNet-50](https://pytorch.org/vision/stable/models.html) pretrained on ImageNet, with [MTCNN](https://github.com/ipazc/mtcnn) for face alignment and cropping. We thank the creators of the [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) and [MAHNOB-HCI](https://mahnob-db.eu/hci-tagging/) datasets for making their data publicly available.
