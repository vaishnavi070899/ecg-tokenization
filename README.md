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

### Experiment 5 — Data Scaling (1K → 2K → 5K → 10K)

**Objective:** Determine whether the codebook collapse observed in Experiment 1 was caused by insufficient training data rather than architectural or hyperparameter limitations.

**Hypothesis:** Training on progressively larger subsets of PTB-XL will naturally increase active code count and improve reconstruction, as the model encounters more diverse signal patterns — even without architectural changes.

**Settings:** K=512, EMA_DECAY=0.95, β=0.25, K-Means Centroid Reset, Adam (LR=1e-3), batch size=32, lead I, z-score normalised. Epoch budget scaled with dataset size.

#### 5.1 Summary

| N (train) | Epochs | Best Val Recon | Mean MSE | Perplexity | Active Codes | Dead Codes | Stability |
|---|---|---|---|---|---|---|---|
| 1K | 20 | 0.0447 | 0.0500 | 355.0 | 375 / 512 | 26.8% | Stable |
| 2K | 25 | 0.0356 | 0.0415 | 291.0 | 363 / 512 | 29.1% | Very stable |
| 5K | 30 | 0.0319 | 0.0376 | 334.0 | 401 / 512 | 21.7% | Stable, mild late noise |
| 10K | 30 | 0.0309 | 0.0354 | 324.5 | 398 / 512 | 22.3% | Slightly noisy |

#### 5.2 Training Dynamics

| N (train) | Convergence Epoch | Final Norm. Perplexity | Dead Codes | Stability |
|---|---|---|---|---|
| 1K | ~17–20 | ~0.69 | 26.8% | Stable |
| 2K | ~20–25 | ~0.65 | 29.1% | Very stable |
| 5K | ~13–17 | ~0.69 | 21.7% | Stable (late noise) |
| 10K | ~15–20 | ~0.70 | 22.3% | Slightly noisy |

#### 5.3 Observations

**1. Reconstruction performance** improves monotonically with dataset size. Best val recon fell from 0.0447 (1K) to 0.0309 (10K) — a 31% reduction. Gains are largest from 1K → 2K and diminish at higher scales, consistent with diminishing returns.

**2. Convergence speed** increases with dataset size. The 1K run needed ~20 epochs to plateau; 5K and 10K converged by ~13–20 epochs. Larger datasets provide more diverse examples per epoch, enabling the model to learn structure more efficiently per pass.

**3. Training stability** is high across all runs. Reconstruction loss decreased smoothly, with train and val curves closely aligned throughout, indicating no overfitting. Minor fluctuations appeared in later epochs for 5K and 10K, reflecting convergence plateaus rather than instability.

**4. Codebook utilisation** remained largely unchanged. Normalised perplexity stayed in the narrow range ~0.65–0.70 across all dataset sizes, and dead-code rates held at ~22–29%. Scaling data alone did not meaningfully reduce collapse — the active-code ceiling appears to be set by training dynamics and the reconstruction objective, not by data diversity.

**5. VQ loss** followed a consistent pattern across all runs: rose during early epochs as the encoder adapted to the discrete bottleneck, then stabilised. No divergence or instability was observed in the quantisation process at any scale.

**Key takeaway:** Scaling the dataset improves reconstruction quality and learning efficiency without compromising stability. However, codebook collapse is not resolved by more data — dead-code rates are nearly identical at 1K and 10K. This confirms the conclusion from Experiments 1–4: collapse is a training-dynamics problem driven by the reconstruction objective's indifference to code diversity, and will require changes to the quantisation mechanism (e.g. entropy regularisation, diversity-aware loss terms) rather than additional data.

---

### Experiment 6 — Residual Vector Quantization (1 → 2 → 3 → 4 stages)

**Objective:** Determine whether stacking multiple VQ layers — each quantizing the residual of the previous — resolves codebook collapse and produces a richer, more diverse token vocabulary than a single codebook.

**Hypothesis:** RVQ will increase effective vocabulary size by distributing the representational burden across multiple codebook layers, preventing any single codebook from collapsing. Each layer will capture progressively finer-grained structure left unrepresented by the previous one.

**Settings:** K=512, EMA_DECAY=0.95, β=0.25, K-Means Centroid Reset, Adam (LR=1e-3), batch size=32, 1K records, 20 epochs, lead I, z-score normalised.

#### 6.1 Per-Stage Results (K=512)

| Config | Stage | Residual MSE | Δ Residual (%) | Mean MSE | Active Codes | Dead Codes | Perplexity |
|---|---|---|---|---|---|---|---|
| VQ-VAE (1 stage) | — | 0.167516 | — | 0.0526 | 377 / 512 | 135 (26.4%) | 309.4 |
| RVQ – 2 layer | 1 | 0.076229 | — | — | 375 / 512 | 137 (26.8%) | 304.4 |
| RVQ – 2 layer | 2 | 0.029856 | −60.8% | 0.0194 | 368 / 512 | 144 (28.1%) | 305.2 |
| RVQ – 3 layer | 1 | 0.045625 | — | — | 378 / 512 | 134 (26.2%) | 308.0 |
| RVQ – 3 layer | 2 | 0.018991 | −58.4% | — | 361 / 512 | 151 (29.5%) | 297.6 |
| RVQ – 3 layer | 3 | 0.010240 | −46.1% | 0.0099 | 395 / 512 | 117 (22.9%) | 327.3 |
| RVQ – 4 layer | 1 | 0.033526 | — | — | 349 / 512 | 163 (31.8%) | 277.1 |
| RVQ – 4 layer | 2 | 0.014834 | −55.8% | — | 357 / 512 | 155 (30.3%) | 291.7 |
| RVQ – 4 layer | 3 | 0.008198 | −44.7% | — | 387 / 512 | 125 (24.4%) | 318.5 |
| RVQ – 4 layer | 4 | 0.005025 | −38.7% | 0.0063 | 398 / 512 | 114 (22.3%) | 332.9 |

#### 6.2 Reconstruction Quality Summary

| Config | Best Val Recon | Mean MSE |
|---|---|---|
| VQ-VAE (1 stage) | 0.0464 | 0.0526 |
| RVQ – 2 layer | 0.0172 | 0.0194 |
| RVQ – 3 layer | 0.0092 | 0.0099 |
| RVQ – 4 layer | 0.0061 | 0.0063 |

Each additional stage yields a ~2–2.5× improvement in MSE. The 1→2 layer jump is the most dramatic (val MSE drops 63%). Beyond 3 layers, gains narrow.

#### 6.3 Codebook Size Sweep (4-layer RVQ, K ∈ {128, 256, 512})

| Config | Best Val Recon | Mean MSE | Avg Dead Codes |
|---|---|---|---|
| K=512, 4 layer | 0.0061 | 0.0063 | ~27% |
| K=256, 4 layer | 0.0074 | 0.0083 | ~16% |
| K=128, 4 layer | 0.0103 | 0.0115 | ~6% |

**K=256 per-stage breakdown:**

| Stage | Residual MSE | Δ Residual (%) | Active Codes | Dead Codes | Perplexity |
|---|---|---|---|---|---|
| 1 | 0.043415 | — | 214 / 256 | 42 (16.4%) | 155.5 |
| 2 | 0.019867 | −54.2% | 210 / 256 | 46 (18.0%) | 152.4 |
| 3 | 0.011128 | −44.0% | 211 / 256 | 45 (17.6%) | 165.0 |
| 4 | 0.006922 | −37.8% | 226 / 256 | 30 (11.7%) | 177.9 |

#### 6.4 Observations

**1. Reconstruction quality** improves monotonically with depth. Val recon drops from 0.046 (1 stage) → 0.017 (2 stages) → 0.009 (3 stages) → 0.006 (4 stages), roughly halving with each additional stage. This confirms RVQ's core premise: each stage captures what the prior stage missed.

**2. Residual compression efficiency degrades with depth.** Stage 1 always captures the largest share of variance. By Stage 4, the per-stage residual gain (0.005) is modest compared to Stage 1 (0.034–0.168). The Δ residual also shrinks monotonically (~55–60% at Stage 2 → ~39% at Stage 4), indicating diminishing returns. 3–4 stages is likely the practical ceiling for this signal complexity.

**3. Dead codes are persistent and depth-independent.** All K=512 configs show ~22–32% dead codes regardless of stage count. This is a codebook utilisation ceiling that RVQ alone cannot solve — it points to a fundamental clustering mismatch between K=512 and the intrinsic cluster structure of the ECG signal, not a training instability.

**4. Perplexity declines with more stages (K=512).** Stage 1 perplexity peaks at ~380–400 for shallow configs, but the overall perplexity falls as depth increases (309 → 277 for 4-layer). Deeper models spread variance across stages, so no single codebook fills as efficiently — Stage 1 is tasked with a harder, more compressed residual distribution.

**5. Codebook size interacts with depth.** K=128 nearly eliminates dead codes (~6%) but caps expressiveness — Mean MSE is 0.0115 vs. 0.0063 for K=512. K=256 is the best tradeoff: ~12–18% dead codes and Mean MSE of 0.0083. This reveals that ~110–120 codes is approximately the natural granularity per RVQ stage for this encoder — K=512 is asking the model to partition a space that supports far fewer separable clusters.

**6. Training dynamics** were stable across all configs. All runs showed fast initial descent (epochs 1–5) followed by monotonic improvement. No divergence or instability was observed at any depth.

**Key takeaway:** 3-layer RVQ offers the best MSE-per-stage efficiency. 4 layers is worthwhile if compute allows, but the marginal gain (0.009 → 0.006 MSE) should be weighed against the persistent dead code problem. The dead code issue is unresolved across all RVQ depths and codebook sizes — it warrants a dedicated intervention (entropy regularisation, usage penalty, or diversity-aware loss) before scaling further. Reducing K is not a full substitute: it forces utilisation but caps capacity. These are complementary approaches.

---

## References

- van den Oord et al. (2017) — [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937)
- Wagner et al. (2020) — [PTB-XL, a large publicly available ECG dataset](https://www.nature.com/articles/s41597-020-0495-6)
- PhysioNet — [PTB-XL Dataset](https://physionet.org/content/ptb-xl/1.0.3/)
