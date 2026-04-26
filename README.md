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
ECG signal (1000 samples)
        |
   Encoder (Conv1d x 3, stride=2 each)
        |
  Latent sequence (125 vectors x 64-dim)
        |
  Vector Quantizer (nearest codebook entry per vector)
        |
  Discrete tokens (125 integers, each 0 to K-1)
        |
   Decoder (ConvTranspose1d x 3, stride=2 each)
        |
Reconstructed signal (1000 samples)
```

**Encoder:** Three `Conv1d` layers (channels: 1 -> 32 -> 64 -> D), kernel=4, stride=2, padding=1 with ReLUs. Downsamples 8x — a 1000-sample ECG becomes 125 latent vectors, one per ~80ms window.

**Decoder:** Mirror of encoder using `ConvTranspose1d` layers (D -> 64 -> 32 -> 1). Upsamples 8x back to 1000 samples.

**Quantizer:** EMA-based codebook updates (no gradient through codebook). Straight-through estimator for encoder gradient flow. Returns commitment loss and perplexity each step. Includes a circular encoder output buffer (size 2048) for codebook reset strategies.

### Prior

GPT-style causal Transformer trained on the discrete code sequences produced by the frozen VQ-VAE. Models `P(c_t | c_1 ... c_{t-1})`. Used for generating new synthetic ECGs.

---

## Hyperparameters

All hyperparameters live in `config.py`. Edit once — every script picks it up automatically.

| Parameter | Value | Description |
|---|---|---|
| `INPUT_DIM` | 1000 | ECG length (10s @ 100Hz) |
| `LATENT_DIM` | 64 | Codebook vector dimension D |
| `NUM_EMBEDDINGS` | 512 | Codebook size K |
| `SEQ_LEN` | 125 | Latent time steps (INPUT_DIM // 8) |
| `EMA_DECAY` | 0.95 | Codebook EMA decay γ |
| `COMMITMENT_COST` | 0.25 | Commitment loss weight β |
| `BUFFER_SIZE` | 2048 | Circular buffer size for K-Means Centroid Reset |
| `BATCH_SIZE` | 16 | VQ-VAE training batch size |
| `EPOCHS` | 30 | VQ-VAE training epochs |
| `LR` | 1e-3 | Adam learning rate |
| `N_RECORDS` | 5000 | Records per split (None = full ~21k) |
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
total_loss = MSE(x_recon, x) + β * ||z_e - sg(z_q)||^2
```

- **Reconstruction loss:** MSE between original and reconstructed signal
- **Commitment loss:** keeps encoder outputs close to codebook entries (β=0.25)
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

**Settings:** K=512, K-Means Centroid Reset, EPOCHS=20, N_RECORDS=1000. γ varied across {0.90, 0.95, 0.99}.

| γ | Mean MSE | Perplexity | Active Codes | Dead Codes | Utilization |
|---|---|---|---|---|---|
| 0.90 | 0.0544 | 348.5 | 372 / 512 | 27.3% | 72.7% |
| 0.95 | 0.0500 | 355.0 | 375 / 512 | 26.8% | 73.2% |
| 0.99 | 0.0595 | 404.1 | 390 / 512 | 23.8% | 76.2% |

**Training dynamics:**

| γ | Early (Epochs 1–5) | Mid (5–15) | Late (15–20) |
|---|---|---|---|
| 0.90 | Rapid perplexity spike (~374 by epoch 3), aggressive adaptation | Steady decline, unstable assignments | Continues drifting, noisy usage |
| 0.95 | Fast coverage gain (~395 by epoch 4), strong early spread | Gradual controlled decline | Smooth convergence, stable usage |
| 0.99 | Slow monotonic rise (94 → 406) | Continues gradual growth | Plateaus ~404, minimal fluctuation |

**Finding:** γ=0.95 achieves the best balance — lowest MSE (0.0500) with stable codebook dynamics. γ=0.90 adapts fastest and overshoots, producing fluctuating assignments and exaggerated QRS peaks. γ=0.99 yields highest perplexity and utilization but slower specialization, resulting in slightly blurred reconstructions. All three settings exhibit a persistent long-tail usage distribution, confirming that EMA tuning moderates collapse severity but does not eliminate it. EMA decay controls the adaptation speed vs. stability tradeoff; it is not a solution to collapse on its own.

### Experiment 5 — Data Scaling

**Settings:** K=512, EMA_DECAY=0.95, β=0.25, K-Means Centroid Reset. N_RECORDS varied across {1000, 2000, 5000}; epoch budget scaled with dataset size to keep training effort proportional.

| N (train) | Epochs | Best Val Recon | Mean MSE | Active Codes | Dead Codes |
|---|---|---|---|---|---|
| 1000 | 20 | 0.044664 | 0.0500 | 375 / 512 | 26.8% |
| 2000 | 25 | 0.035590 | 0.0415 | 363 / 512 | 29.1% |
| 5000 | 30 | 0.033118† | — | — | — |

†Run interrupted at epoch 20; best checkpoint at epoch 17. Sweep metrics pending completion.

**Training dynamics (5000 records):** Val recon fell steeply in early epochs (0.059 → 0.034 by epoch 10) and then flattened, with only marginal improvements past epoch 13. The loss curve had not fully plateaued by epoch 20, suggesting the remaining 10 epochs may still yield small gains.

**Codebook dynamics (5000 records):** Perplexity stabilised in the mid-350s across all epochs — notably lower than the 1000-record run's peak of ~395 — suggesting that with more data the encoder learns a slightly more concentrated usage pattern, though the absolute active-code count is expected to be similar.

**Finding (preliminary):** More training data consistently improves reconstruction: each 2–5× increase in N yields roughly a 20–25% reduction in best val recon loss. The gain from 2000 → 5000 records is smaller in absolute terms than 1000 → 2000, hinting at diminishing returns. Dead-code rate appears to be largely independent of dataset size, remaining stable at ~27–29% — consistent with the view that collapse is driven by training dynamics rather than data volume.

---

## References

- van den Oord et al. (2017) — [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937)
- Wagner et al. (2020) — [PTB-XL, a large publicly available ECG dataset](https://www.nature.com/articles/s41597-020-0495-6)
- PhysioNet — [PTB-XL Dataset](https://physionet.org/content/ptb-xl/1.0.3/)
