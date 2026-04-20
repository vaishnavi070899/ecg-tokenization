"""extract_codes.py

Run the frozen VQ-VAE encoder+quantizer over the full PTB-XL dataset once and
save the resulting code index sequences to disk.

Outputs
-------
codes_train.npy  -- shape (N_train, 125) int64
codes_val.npy    -- shape (N_val,   125) int64

These files are the input to train_prior.py. Extracting once avoids re-running
the encoder on every prior training batch.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data.load_data import PTBXLDataset
from models.vqvae import VQVAE

CHECKPOINT = "vqvae_best.pt"
BATCH_SIZE = 64   # larger is fine here -- no gradients

# ── Load model ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = VQVAE(input_dim=config.INPUT_DIM, latent_dim=config.LATENT_DIM,
               num_embeddings=config.NUM_EMBEDDINGS)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.to(device).eval()
print(f"Loaded checkpoint: {CHECKPOINT}")


def extract(folds, label):
    dataset   = PTBXLDataset(folds=folds)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_codes = []

    for i, x in enumerate(loader):
        x       = x.to(device)
        indices = model.encode_indices(x)          # (B, SEQ_LEN) int64
        all_codes.append(indices.cpu().numpy())

        n_done = min((i + 1) * BATCH_SIZE, len(dataset))
        if (i + 1) % 20 == 0 or n_done == len(dataset):
            print(f"  {label}: {n_done}/{len(dataset)} records", end="\r")

    codes = np.concatenate(all_codes, axis=0)      # (N, SEQ_LEN)
    path  = f"codes_{label}.npy"
    np.save(path, codes)
    print(f"\n  Saved {codes.shape} -> {path}")
    return codes


print("Extracting train codes (folds 1-9)...")
extract(folds=list(range(1, 10)), label="train")

print("Extracting val codes   (fold 10)...")
extract(folds=[10], label="val")

print("Done.")
