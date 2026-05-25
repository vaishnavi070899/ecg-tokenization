# config.py — single source of truth for all hyperparameters

import os

# ── Dataset ────────────────────────────────────────────────────────────────────
PTB_XL_PATH = os.environ.get(
    "PTB_XL_PATH",
    "C:/Users/vaish/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
)

# ── Signal ─────────────────────────────────────────────────────────────────────
INPUT_DIM      = 1000   # samples per ECG (10 s at 100 Hz)
SAMPLING_RATE  = 100    # Hz
LEAD           = 0      # lead index (0 = lead I)

# ── VQ-VAE architecture ────────────────────────────────────────────────────────
LATENT_DIM      = 64   # encoder output channels = codebook vector dimension D
NUM_EMBEDDINGS           = 256              # codebook size K for single-stage VQ
NUM_EMBEDDINGS_PER_STAGE = [256, 256, 128, 64]  # per-stage K for RVQ (index = stage - 1)
NUM_RVQ_STAGES  = 4    # RVQ stages: 1 = standard single-stage VQ, 2+ = residual VQ
SEQ_LEN         = INPUT_DIM // 8   # 125 — latent time steps after 8x downsampling

# ── VQ-VAE training ────────────────────────────────────────────────────────────
EMA_DECAY        = 0.95   # EMA decay γ for codebook updates
COMMITMENT_COST  = 0.25   # β — weight on commitment loss
BUFFER_SIZE      = 2048   # circular buffer size for K-Means Centroid Reset (Strategy 2)
BATCH_SIZE       = 32
EPOCHS           = 30
LR               = 1e-3
N_RECORDS        = 2000   # cap per split for quick runs; set to None for full ~21k dataset

# ── NS-VQ (Non-Stationary Vector Quantization) ────────────────────────────────
NSVQ_ALPHA  = 0.1    # kernel update step size for non-selected codes
# Per-stage τ (bandwidth) = scale of RBF kernel; derived from mean NN distances
# Stage:                    1      2      3      4
NSVQ_TAU_A  = [0.40,  0.23,  0.14,  0.10]   # tight   (×0.2 × mean NN dist)
NSVQ_TAU_B  = [2.01,  1.13,  0.71,  0.49]   # moderate (×1.0 × mean NN dist)
NSVQ_TAU_C  = [4.03,  2.25,  1.41,  0.98]   # broad    (×2.0 × mean NN dist)
NSVQ_TAU_D  = [1.75,  1.05,  0.69,  0.48]   # based on run-B distances
NSVQ_REPULSE_GAMMA  = 0.05   # step size for repulsion away from local neighbourhood
NSVQ_AT_RISK_THRESH = 5.0    # only apply NS-VQ to codes with EMA usage below this
# ── Prior architecture ─────────────────────────────────────────────────────────
D_MODEL   = 128
N_HEADS   = 4
N_LAYERS  = 4
DROPOUT   = 0.1

# ── Prior training ─────────────────────────────────────────────────────────────
PRIOR_BATCH_SIZE  = 128
PRIOR_EPOCHS      = 50
PRIOR_LR          = 1e-3
