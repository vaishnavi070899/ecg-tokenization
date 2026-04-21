# config.py — single source of truth for all hyperparameters

import os

# ── Dataset ────────────────────────────────────────────────────────────────────
PTB_XL_PATH = os.environ.get(
    "PTB_XL_PATH",
    "path"
)

# ── Signal ─────────────────────────────────────────────────────────────────────
INPUT_DIM      = 1000   # samples per ECG (10 s at 100 Hz)
SAMPLING_RATE  = 100    # Hz
LEAD           = 0      # lead index (0 = lead I)

# ── VQ-VAE architecture ────────────────────────────────────────────────────────
LATENT_DIM      = 64   # encoder output channels = codebook vector dimension D
NUM_EMBEDDINGS  = 512   # codebook size K
SEQ_LEN         = INPUT_DIM // 8   # 125 — latent time steps after 8x downsampling

# ── VQ-VAE training ────────────────────────────────────────────────────────────
EMA_DECAY        = 0.95   # EMA decay γ for codebook updates
COMMITMENT_COST  = 0.25   # β — weight on commitment loss
BUFFER_SIZE      = 2048   # circular buffer size for K-Means Centroid Reset (Strategy 2)
BATCH_SIZE       = 32
EPOCHS           = 50
LR               = 1e-3
N_RECORDS        = 5000   # cap per split for quick runs; set to None for full ~21k dataset

# ── Prior architecture ─────────────────────────────────────────────────────────
D_MODEL   = 128
N_HEADS   = 4
N_LAYERS  = 4
DROPOUT   = 0.1

# ── Prior training ─────────────────────────────────────────────────────────────
PRIOR_BATCH_SIZE  = 128
PRIOR_EPOCHS      = 50
PRIOR_LR          = 1e-3
