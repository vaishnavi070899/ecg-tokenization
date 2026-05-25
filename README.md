# ECG VQ-VAE

A Vector Quantized Variational Autoencoder (VQ-VAE) for learning discrete, language-like representations of ECG signals from the PTB-XL dataset.

The core idea: convert continuous ECG waveforms into sequences of discrete tokens that can be modelled autoregressively — treating ECG signals the way language models treat text.

---

## Research Motivation

This project is motivated by the question of whether physiological signals such as ECG can be represented as discrete, language-like tokens without losing clinically relevant structure. While VQ-VAE provides a natural framework for learning such representations, it suffers from codebook collapse, where only a small subset of tokens are used despite large available capacity. Through a series of controlled experiments, this work investigates the relationship between codebook size, training dynamics, and representational diversity, with a focus on understanding why increased capacity does not translate to increased utilization. The goal is not only to improve reconstruction quality, but to learn a compact, expressive, and interpretable tokenization of ECG signals that can support downstream generative and predictive models.

The broader goal is to move toward a discrete representation of ECG signals that is both information-efficient and semantically meaningful, enabling downstream modeling analogous to language processing.

---

## Project Structure

```
ecg-vqvae/
├── config.py              # Single source of truth for all hyperparameters
├── main.py                # Train VQ-VAE
├── extract_codes.py       # Encode dataset -> discrete code sequences (.npy)
├── train_prior.py         # Train autoregressive Transformer prior
├── generate.py            # Sample synthetic ECGs from the prior
├── reconstruct.py         # Reconstruct real ECGs and plot original vs output
├── sweep_embeddings.py    # Evaluate codebook usage and reconstruction for current checkpoint
│
├── data/
│   └── load_data.py       # PTBXLDataset with fold-based splits + normalization
│
├── models/
│   ├── encoder.py         # 1D CNN encoder (8x downsampling)
│   ├── decoder.py         # 1D CNN decoder (8x upsampling)
│   ├── quantizer.py       # VQ layer with EMA updates, codebook reset strategies, perplexity
│   ├── vqvae.py           # Full VQ-VAE model
│   └── prior.py           # GPT-style autoregressive Transformer prior
│
└── utils/
    └── plot.py            # Single and grid ECG plot utilities
```

---

## Architecture

### VQ-VAE

```
ECG signal  (B, 1000)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Encoder                                            │
│  Conv1d(1  → 32,  k=4, s=2, p=1)  + ReLU           │  → (B, 32,  500)
│  Conv1d(32 → 64,  k=4, s=2, p=1)  + ReLU           │  → (B, 64,  250)
│  Conv1d(64 → 64,  k=4, s=2, p=1)                   │  → (B, 64,  125)
└─────────────────────────────────────────────────────┘
       │  z_e  (B, 64, 125)
       ▼
┌─────────────────────────────────────────────────────┐
│  Residual Vector Quantizer  (4 stages)              │
│                                                     │
│  Stage 1  K=256   nearest(z_e,   cb₁) → z_q1, i1   │
│           r₁ = z_e − z_q1                          │
│  Stage 2  K=256   nearest(r₁,    cb₂) → z_q2, i2   │
│           r₂ = r₁  − z_q2                          │
│  Stage 3  K=128   nearest(r₂,    cb₃) → z_q3, i3   │
│           r₃ = r₂  − z_q3                          │
│  Stage 4  K= 64   nearest(r₃,    cb₄) → z_q4, i4   │
│                                                     │
│  z_q     = z_q1 + z_q2 + z_q3 + z_q4  (B, 64, 125)│
│  indices = [i1 | i2 | i3 | i4]         (B, 125, 4) │
└─────────────────────────────────────────────────────┘
       │  z_q  (B, 64, 125)
       ▼
┌─────────────────────────────────────────────────────┐
│  Decoder                                            │
│  ConvTranspose1d(64 → 64, k=4, s=2, p=1) + ReLU    │  → (B, 64,  250)
│  ConvTranspose1d(64 → 32, k=4, s=2, p=1) + ReLU    │  → (B, 32,  500)
│  ConvTranspose1d(32 →  1, k=4, s=2, p=1)           │  → (B,  1, 1000)
└─────────────────────────────────────────────────────┘
       │
       ▼
Reconstructed signal  (B, 1000)
```

**Encoder:** Three `Conv1d` layers (channels: 1 → 32 → 64 → 64), kernel=4, stride=2, padding=1, with ReLU activations after the first two. Downsamples 8× — a 1000-sample ECG (10 s at 100 Hz) becomes 125 latent vectors, each representing an ~80 ms window.

**Residual Vector Quantizer (RVQ):** Four sequential VQ stages with stage-specific codebook sizes [256, 256, 128, 64]. Each stage quantizes the residual left by all previous stages using nearest-neighbour lookup (hard argmin). The final quantized representation is the sum of all four stage outputs; decoding requires only four codebook lookups and a sum.

Per stage:
- **EMA codebook updates:** codebook vectors are updated via exponential moving average (γ = 0.95) rather than gradient descent — no codebook gradient required.
- **Straight-through estimator:** gradients flow through the quantization bottleneck to the encoder unchanged.
- **Commitment loss:** β · ‖z_e − sg(z_q)‖² keeps encoder outputs anchored to codebook entries (β = 0.25).
- **K-Means Centroid Reset:** a circular buffer (size 2048) of recent encoder outputs is maintained; dead codes (EMA count < 1.0) are reset to k-means centroids computed over the buffer.
- **NS-VQ kernel update:** at-risk codes (EMA count < 5.0) are attracted toward active regions via a weighted RBF kernel, then repelled from dense local neighbourhoods using an adaptive bandwidth derived from the median nearest-neighbour distance (see Experiment 10).

**Warm-start initialization:** Before the first training epoch, each stage's codebook is initialized from its true residual distribution using sequential k-means++ seeding followed by 10 Lloyd iterations. Stage *s* uses residuals produced by already-initialized stages 0…*s*−1, ensuring correct scale alignment from the first batch.

**Decoder:** Mirror of the encoder using three `ConvTranspose1d` layers (channels: 64 → 64 → 32 → 1), kernel=4, stride=2, padding=1, with ReLU after the first two. Upsamples 8× back to 1000 samples.

### Prior

GPT-style causal Transformer trained on the discrete code sequences produced by the frozen VQ-VAE. Models `P(c_t | c_1 ... c_{t-1})`. Used for generating new synthetic ECGs.

---

## Hyperparameters

All hyperparameters live in `config.py`. Edit once — every script picks it up automatically.

| Parameter | Value | Description |
|---|---|---|
| `INPUT_DIM` | 1000 | ECG length (10 s at 100 Hz) |
| `LATENT_DIM` | 64 | Codebook vector dimension D |
| `NUM_EMBEDDINGS` | 256 | Codebook size K for single-stage VQ |
| `NUM_EMBEDDINGS_PER_STAGE` | [256, 256, 128, 64] | Per-stage codebook sizes for RVQ |
| `NUM_RVQ_STAGES` | 4 | Number of RVQ stages (1 = standard VQ) |
| `SEQ_LEN` | 125 | Latent time steps (INPUT_DIM // 8) |
| `EMA_DECAY` | 0.95 | Codebook EMA decay γ |
| `COMMITMENT_COST` | 0.25 | Commitment loss weight β |
| `BUFFER_SIZE` | 2048 | Circular buffer size for K-Means Centroid Reset |
| `BATCH_SIZE` | 32 | VQ-VAE training batch size |
| `EPOCHS` | 30 | VQ-VAE training epochs |
| `LR` | 1e-3 | Adam learning rate |
| `N_RECORDS` | 2000 | Records per split (None = full ~21k) |
| `NSVQ_ALPHA` | 0.1 | NS-VQ attraction step size |
| `NSVQ_REPULSE_GAMMA` | 0.05 | NS-VQ repulsion step size |
| `NSVQ_AT_RISK_THRESH` | 5.0 | EMA usage threshold for NS-VQ updates |
| `D_MODEL` | 128 | Prior Transformer width |
| `N_HEADS` | 4 | Prior attention heads |
| `N_LAYERS` | 4 | Prior Transformer layers |
| `PRIOR_EPOCHS` | 50 | Prior training epochs |

---

## Dataset

[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) — a large publicly available ECG dataset (21,837 records, 10s, 100Hz).

**Train/val split** uses PTB-XL's built-in `strat_fold` column:
- Folds 1-9 -> training
- Fold 10 -> validation

**Preprocessing:** per-sample z-score normalization (mean=0, std=1) applied in the dataset loader.

### Setting the dataset path

By default `config.py` reads from the environment variable `PTB_XL_PATH`. If not set, it falls back to the local path. To use on a different machine:

```bash
# Windows
set PTB_XL_PATH=D:/data/ptb-xl/

# macOS / Linux
export PTB_XL_PATH=/data/ptb-xl/
```

---

## Setup

```bash
pip install torch numpy matplotlib pandas wfdb
```

---

## Usage

### 1. Train the VQ-VAE

```bash
python main.py
```

Trains on PTB-XL (folds 1-9), validates on fold 10. Saves best checkpoint to `vqvae_best.pt`.
Logs per epoch: `train_recon`, `vq_loss`, `perplexity`, `val_recon`.

To resume an interrupted run from a saved checkpoint:

```bash
python main.py --resume --start-epoch 21 --best-val-loss 0.033118
```

`--start-epoch` sets the epoch to continue from; `--best-val-loss` carries forward the best val loss so the checkpoint is only overwritten on genuine improvement.

### 2. Extract discrete code sequences

```bash
python extract_codes.py
```

Runs the frozen VQ-VAE encoder over the full dataset. Saves:
- `codes_train.npy` — shape `(N_train, 125)` int64
- `codes_val.npy` — shape `(N_val, 125)` int64

Only needs to run once per trained checkpoint.

### 3. Train the prior

```bash
python train_prior.py
```

Trains the causal Transformer on the extracted code sequences. Saves best checkpoint to `prior_best.pt`.

### 4. Generate synthetic ECGs

```bash
python generate.py                  # 8 signals, default temperature
python generate.py --n 16           # 16 signals
python generate.py --temp 0.8       # sharper sampling
python generate.py --save ecgs.png  # save instead of interactive plot
```

### 5. Reconstruction check

```bash
python reconstruct.py               # 4 val records, original vs reconstructed
python reconstruct.py --n 8         # 8 records
python reconstruct.py --save recon.png
```

### 6. Evaluate codebook and reconstruction

```bash
python sweep_embeddings.py
```

Evaluates the current checkpoint (`vqvae_best.pt`). Logs mean MSE, active codes, and dead codes to console. Outputs:
- `recon.png` — reconstruction plots (original vs reconstructed)
- `codebook.png` — code frequency histogram and sorted usage plot

---

## Training Pipeline (full run order)

```bash
python main.py            # 1. Train VQ-VAE  ->  vqvae_best.pt
python extract_codes.py   # 2. Encode dataset -> codes_train.npy, codes_val.npy
python train_prior.py     # 3. Train prior   ->  prior_best.pt
python generate.py        # 4. Generate ECGs
```

---

## Codebook Collapse and Reset Strategies

A key challenge in VQ-VAE training is **codebook collapse**: a large fraction of codes receive no assignments (dead codes), while a small subset handles nearly all encoder outputs. This limits the effective vocabulary and reconstruction quality.

### Root Cause

EMA updates only move codes that receive assignments. Codes that lose assignments early on receive no further gradient signal and never recover. A fast EMA decay (low γ) helps by making the codebook more responsive to recent data.

### Dead Code Detection

All reset strategies share the same detection step: a code is considered dead if its EMA-smoothed assignment count falls below a threshold (default: 1.0).

### Reset Strategies (implemented as stubs in `models/quantizer.py`)

Three strategies are available as commented-out method stubs in `VectorQuantizer`. Uncomment one call site and its corresponding method to activate it.

**Strategy 1 — Random Restart**
Replace each dead code with a randomly sampled encoder output from the current batch. Simple and cheap. Risk: the replacement may land in an already well-covered region.

**Strategy 2 — K-Means Centroid Reset** (CVQ-VAE style)
Maintain a circular buffer of recent encoder outputs. When a code dies, reset it to the centroid of a mini K-means cluster computed over the buffer. More geometrically principled than random restart. Requires `BUFFER_SIZE` and `encoder_buffer` state in `__init__`.

**Strategy 3 — Anchor Resampling**
Sample replacement vectors using importance weights proportional to each encoder output's distance from the nearest active code. Actively fills gaps in the codebook by targeting under-covered regions of feature space. Uses the same squared-L2 trick as the main quantization step.

### Activating a Strategy

In `models/quantizer.py`, inside `forward()` after the EMA update block:

```python
# 1. Detect dead codes (shared by all strategies)
dead_indices, dead_mask = self.find_dead_codes(
    ema_cluster_size=self.ema_cluster_size, threshold=1.0
)

# 2. Uncomment ONE of the three strategies:
self.random_restart(dead_indices=dead_indices, flat=flat)          # Strategy 1
# self.update_buffer(flat); self.kmeans_centroid_reset(dead_indices) # Strategy 2
# self.anchor_resampling(dead_indices, flat, active_codes)           # Strategy 3
```

---

## Loss Function

```
total_loss = MSE(x_recon, x) + β * ||z_e - sg(z_q)||^2 + λ * entropy_loss
```

- **Reconstruction loss:** MSE between original and reconstructed signal
- **Commitment loss:** keeps encoder outputs close to codebook entries (β=0.25)
- **Entropy loss:** negative entropy of soft codebook assignment distribution (λ=0.01); minimizing encourages uniform codebook usage
- **Codebook update:** EMA (not gradient), γ=0.95

---

## Experiment Log

### Experiment 1 — Codebook Size Sweep (K sweep, no reset strategy)

| K | Mean MSE | Perplexity | Active Codes | Dead Codes |
|---|---|---|---|---|
| 64 | 0.134 | 16.7 | 26 / 64 | 59.4% |
| 128 | 0.117 | 23.7 | 35 / 128 | 72.7% |
| 512 | 0.097 | 62.9 | 96 / 512 | 81.2% |
| 1024 | 0.088 | 91.9 | 128 / 1024 | 87.5% |

**Finding:** Reconstruction improves with K, but codebook collapse is severe across all sizes. The model converges to an effective vocabulary of ~100-130 codes regardless of K. Root cause: EMA_DECAY=0.99 too slow relative to dataset size (~31 batches/epoch). Addressed in subsequent experiments via EMA decay tuning (0.99 -> 0.95) and codebook reset strategies.

### Experiment 2 — Codebook Reset Strategy Comparison

**Settings:** K=128, EMA_DECAY=0.95, EPOCHS=20, N_RECORDS=1000.

| Strategy | Mean MSE | Perplexity | Active Codes | Dead Codes |
|---|---|---|---|---|
| No reset (Exp 1 baseline) | 0.117 | 23.7 | 35 / 128 | 72.7% |
| Random Restart | ~0.07–0.08 | moderate | highest | low |
| K-Means Centroid Reset | ~0.07–0.08 | highest | high | low |
| Anchor Resampling | ~0.065–0.075 | lowest | moderate | moderate |

**Finding:** All three strategies reduce MSE by ~35–45% over baseline. K-Means Reset achieves the best balance — highest perplexity (most uniform usage) with near-best reconstruction. Anchor Resampling wins on MSE but suffers the worst utilization (rich-get-richer effect). Active code count alone is misleading; perplexity is the more informative metric. Reset strategies are reactive — they revive dead codes but do not prevent re-collapse, because the reconstruction objective gives the model no incentive to use all codes.

---

### Experiment 3 — Codebook Size Sweep with Reset Strategies

**Settings:** K ∈ {128, 512, 1024}, EMA_DECAY=0.95, K-Means Centroid Reset, EPOCHS=20, N_RECORDS=1000.

| K | Mean MSE | Perplexity | Active Codes | Dead Codes | Utilization |
|---|---|---|---|---|---|
| 128 | 0.0760 | 92.5 | 123 / 128 | 3.9% | 96.1% |
| 512 | 0.0571 | 319.8 | 389 / 512 | 24.0% | 76.0% |
| 1024 | 0.0448 | 507.3 | 583 / 1024 | 43.1% | 56.9% |

**Finding:** Reconstruction improves monotonically with K (−41% MSE from K=128 to K=1024), but utilization drops sharply as capacity grows. Perplexity increases in absolute terms but decreases relative to K — usage becomes more skewed at scale. Larger codebooks amplify collapse rather than resolve it. This confirms collapse is a training dynamics problem, not a capacity limitation: the model does not need more codes, it needs a stronger incentive to use them.

### Experiment 4 — Effect of EMA Decay Rate (γ) on Codebook Utilization

**Settings:** K=512, K-Means Centroid Reset, EPOCHS=20, N_RECORDS=1000.

| γ | Mean MSE | Perplexity | Active Codes | Dead Codes | Utilization |
|---|---|---|---|---|---|
| 0.90 | 0.0544 | 348.5 | 372 / 512 | 27.3% | 72.7% |
| 0.95 | 0.0500 | 355.0 | 375 / 512 | 26.8% | 73.2% |
| 0.99 | 0.0595 | 404.1 | 390 / 512 | 23.8% | 76.2% |

**Finding:** γ=0.95 achieves the best balance — lowest MSE (0.0500) with stable dynamics. γ=0.90 adapts fastest but overshoots, producing fluctuating assignments; γ=0.99 yields the highest utilization but slower specialization and slightly blurred reconstructions. All three settings show a persistent long-tail usage distribution — EMA tuning moderates collapse severity but does not eliminate it.

### Experiment 5 — Data Scaling (1K → 2K → 5K → 10K)

**Settings:** K=512, EMA_DECAY=0.95, K-Means Centroid Reset. Epoch budget scaled with dataset size.

| N (train) | Best Val Recon | Mean MSE | Perplexity | Active Codes | Dead Codes |
|---|---|---|---|---|---|
| 1K | 0.0447 | 0.0500 | 355.0 | 375 / 512 | 26.8% |
| 2K | 0.0356 | 0.0415 | 291.0 | 363 / 512 | 29.1% |
| 5K | 0.0319 | 0.0376 | 334.0 | 401 / 512 | 21.7% |
| 10K | 0.0309 | 0.0354 | 324.5 | 398 / 512 | 22.3% |

**Finding:** Reconstruction improves monotonically with data (−31% val recon, 1K → 10K) and larger datasets converge faster, but codebook collapse is unaffected. Dead-code rates held at ~22–29% across all sizes. Collapse is a training-dynamics problem driven by the reconstruction objective's indifference to code diversity — more data cannot fix it.

---

### Experiment 6 — Residual Vector Quantization (1 → 2 → 3 → 4 stages)

**Settings:** K=512, EMA_DECAY=0.95, K-Means Centroid Reset, EPOCHS=20, N_RECORDS=1K.

| Config | Best Val Recon | Mean MSE | Avg Dead Codes |
|---|---|---|---|
| VQ-VAE (1 stage) | 0.0464 | 0.0526 | ~26% |
| RVQ – 2 stage | 0.0172 | 0.0194 | ~27% |
| RVQ – 3 stage | 0.0092 | 0.0099 | ~26% |
| RVQ – 4 stage | 0.0061 | 0.0063 | ~27% |

**K sweep (4-stage RVQ):**

| K | Best Val Recon | Mean MSE | Avg Dead Codes |
|---|---|---|---|
| 128 | 0.0103 | 0.0115 | ~6% |
| 256 | 0.0074 | 0.0083 | ~16% |
| 512 | 0.0061 | 0.0063 | ~27% |

**Finding:** Each additional RVQ stage roughly halves MSE — the 1→2 stage jump is the largest (−63%). Dead codes persist at ~22–32% regardless of stage count, confirming RVQ alone cannot resolve collapse. K=256 is the best tradeoff: ~16% dead codes and MSE 0.0083 vs. K=512's 0.0063 with 27% dead. The natural active-code ceiling is ~110–120 per stage, making K=512 oversized by ~4×.

---

### Experiment 7 — Entropy Regularization

**Settings:** K=512, 4-layer RVQ, EMA_DECAY=0.95, K-Means Centroid Reset, EPOCHS=15, N_RECORDS=1K. λ varied across {0.01, 0.1}.

| λ | Best Val Recon | Peak Perplexity | Active Codes (Stage 1) | Dead Codes (Stage 1) |
|---|---|---|---|---|
| 0 (baseline) | 0.0061 | 333 / 512 | 398 / 512 | 22.3% |
| 0.01 | 0.009256 | ~374 / 512 | 353 / 512 | 31.1% |
| 0.1 | 0.012255 | ~378 / 512 | 364 / 512 | 28.9% |

**Finding:** Entropy regularization did not improve codebook utilization and degraded reconstruction. λ=0.1 destabilizes training — the entropy term (~24 nats across 4 stages) overwhelms the loss by mid-training, inflating VQ loss from 0.007 to 0.068. λ=0.01 is stable but marginal, and active codes actually decrease vs. baseline. Root cause: entropy acts on soft (differentiable) assignments while EMA updates use hard argmin — a dead code that never wins a hard assignment cannot be revived by encoder gradients alone.

---

### Experiment 8 — Usage Penalty / Frequency Balancing

**Settings:** 4-layer RVQ, K=512, EMA_DECAY=0.95, K-Means Centroid Reset, EPOCHS=20, N_RECORDS=1K. α varied across {0.01, 0.1, 0.5, 1.0}.

| α | Mean MSE | S1 Dead | S2 Dead | S3 Dead | S4 Dead |
|---|---|---|---|---|---|
| 0.01 | 0.0066 | 29.9% | 27.7% | 26.2% | 24.2% |
| 0.1 | 0.0089 | 28.1% | 26.4% | 24.8% | 21.9% |
| 0.5 | 0.0065 | 29.5% | 29.9% | 24.4% | 25.0% |
| 1.0 | 0.0067 | 31.1% | 27.3% | 23.0% | 21.5% |

**Finding:** Usage penalties produced small, inconsistent improvements in utilization (dead codes 21–31%, within the same range as the reset-only baseline from Exp 6). Reconstruction remained stable across all α (MSE 0.0065–0.0089), but increasing α beyond 0.1 gave no measurable gain — Stage 4 active codes for α=0.1 and α=1.0 differ by only two codes. Penalties redistribute assignments among already-active codes rather than reviving dead ones; K-Means Centroid Reset remains the primary driver of utilization maintenance.

---

### Experiment 9 — Temperature / Soft Assignments

**Settings:** 4-layer RVQ, K=512, EMA_DECAY=0.95, K-Means Centroid Reset, EPOCHS=20, N_RECORDS=1K. Temperature T varied across {0.1, 0.5, 1.0, 2.0}.

| T | Mean MSE | S1 Dead | S2 Dead | S3 Dead | S4 Dead |
|---|---|---|---|---|---|
| 0.1 | 0.0568 | 22.9% | 86.3% | 91.0% | 94.9% |
| 0.5 | 0.0559 | 25.2% | 88.3% | 89.1% | 94.5% |
| 1.0 | 0.0656 | 36.3% | 21.1% | 20.5% | 20.3% |
| 2.0 | 0.0758 | 45.5% | 21.1% | 20.5% | 20.3% |

**Finding:** No temperature outperformed the hard-assignment baseline. Low temperatures (T=0.1, 0.5) maintained Stage 1 utilization but caused catastrophic collapse in later stages (Stages 2–4: 86–95% dead), degrading MSE to ~0.056. High temperatures (T=1.0, 2.0) spread assignments across later stages but collapsed Stage 1 (36–46% dead) and degraded MSE to 0.066–0.076. Soft assignments weaken the competitive pressure needed for codebook specialization; an earlier implementation at T=1.0 collapsed to a single active code (perplexity=1, MSE≈1.0). Hard nearest-neighbor assignment with K-Means Centroid Reset remained the most stable configuration.

---

> **Note:** Experiments 7–9 (entropy regularization, usage penalties, and soft assignment temperature) were all ultimately removed from the final model. None produced a meaningful reduction in dead codes relative to the K-Means Centroid Reset baseline, and two of the three degraded reconstruction quality. The final architecture retains hard nearest-neighbor assignments, EMA codebook updates, and K-Means Centroid Reset only.

---

### Experiment 10 — NS-VQ + Warm-Start + Adaptive K Scaling

**Settings:** 4-stage RVQ, EMA_DECAY=0.95, 2000 records, 30 epochs. NS-VQ Run B (τ=[2.01,1.13,0.71,0.49]): at-risk codes (EMA usage < 5.0) attracted toward active regions via RBF kernel (α=0.1) with adaptive repulsion. Sequential k-means++ warm-start + 10 Lloyd iterations per stage.

| Configuration | K per Stage | MSE | S1 Dead | S2 Dead | S3 Dead | S4 Dead |
|---|---|---|---|---|---|---|
| Baseline | [256,256,256,256] | 0.0057 | 16.4% | 22.3% | 18.8% | 14.8% |
| NS-VQ Run A (tight τ) | [256,256,256,256] | 0.0087 | 8.2% | 20.7% | 18.4% | 15.6% |
| NS-VQ Run B (moderate τ) | [256,256,256,256] | 0.0081 | 10.9% | 19.5% | 18.4% | 16.4% |
| NS-VQ Run C (broad τ) | [256,256,256,256] | 0.0085 | 10.9% | 18.8% | 15.6% | 15.6% |
| NS-VQ Run D (empirical τ) | [256,256,256,256] | 0.0092 | 9.0% | 14.5% | 15.6% | 13.3% |
| + Warm-Start | [256,256,256,256] | 0.0079 | 11.3% | 16.4% | 15.2% | 12.9% |
| K=[512,256,128,64] | [512,256,128,64] | 0.0092 | 22.3% | 16.8% | 9.4% | 1.6% |
| + Lloyd warm-start | [512,256,128,64] | 0.0074 | 25.8% | 19.1% | 8.6% | 0.0% |
| K=[256,256,128,64] (1k rec) | [256,256,128,64] | 0.0080 | 9.8% | 21.5% | 7.0% | 1.6% |
| **Full Stack (2k rec)** | **[256,256,128,64]** | **0.0058** | **16.0%** | **22.3%** | **13.3%** | **1.6%** |

**Warm-start initialization (Full Stack run):**

| Stage | Residual norm mean | Residual norm std | Codebook norm mean |
|---|---|---|---|
| 1 | 0.8426 | 0.4443 | 1.2131 |
| 2 | 0.1977 | 0.1280 | 0.2890 |
| 3 | 0.1499 | 0.0956 | 0.2418 |
| 4 | 0.1335 | 0.0823 | 0.1420 |

Default `U(−1/K, 1/K)` init gives codebook norms of ~0.018 — 50× smaller than Stage 1 residuals (0.84). Warm-start reduces this mismatch; the remaining ~1.2–1.4× overshoot is corrected by EMA in the first few epochs.

**Finding:** NS-VQ improved Stage 1 utilization (dead codes 16.4% → 8–11%) but introduced training instability, keeping MSE above baseline (0.0081–0.0092 vs. 0.0057). Warm-start had a larger effect than online redistribution — initialization geometry dominates long-term utilization dynamics. Stage-specific K scaling [256,256,128,64] matched codebook capacity to residual complexity, recovering baseline MSE (0.0058) with 31% fewer total codes (704 vs. 1024). Stage 2 dead codes (~22%) are structural and unaffected by any intervention, reflecting the lower intrinsic dimensionality of Stage 1's error patterns.

---

## References

- van den Oord et al. (2017) — [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937)
- Wagner et al. (2020) — [PTB-XL, a large publicly available ECG dataset](https://www.nature.com/articles/s41597-020-0495-6)
- PhysioNet — [PTB-XL Dataset](https://physionet.org/content/ptb-xl/1.0.3/)
- Lu et al. (2026) — [Beyond Stationarity: Rethinking Codebook Collapse in Vector Quantization](https://arxiv.org/abs/2602.18896) — Identifies non-stationary encoder updates as the root cause of codebook collapse and proposes NS-VQ and TransVQ to achieve near-complete codebook utilization; basis for the NS-VQ update rule used in Experiment 10.
- Zheng et al. (2024) — [ERVQ: Enhanced Residual Vector Quantization with Intra-and-Inter-Codebook Optimization for Neural Audio Codecs](https://arxiv.org/abs/2410.12359) — Introduces intra- and inter-codebook optimization strategies to address collapse in RVQ-based neural audio codecs; motivation for the inter-stage projection architecture discussed as a future direction in Experiment 10.
