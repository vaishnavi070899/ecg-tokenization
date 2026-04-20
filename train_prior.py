"""
train_prior.py

Train the autoregressive Transformer prior over VQ-VAE code sequences.

Expects codes_train.npy and codes_val.npy to already exist (run extract_codes.py first).
Saves the best checkpoint to prior_best.pt.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import config
from models.prior import ECGPrior


# ── Dataset ────────────────────────────────────────────────────────────────────
class CodesDataset(Dataset):
    """Thin wrapper around a (N, seq_len) int64 numpy array of code indices."""

    def __init__(self, path):
        self.codes = torch.from_numpy(np.load(path)).long()  # (N, seq_len)

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        return self.codes[idx]   # (seq_len,)


train_dataset = CodesDataset("codes_train.npy")
val_dataset   = CodesDataset("codes_val.npy")

train_loader  = DataLoader(train_dataset, batch_size=config.PRIOR_BATCH_SIZE, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=config.PRIOR_BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_dataset)} sequences  |  Val: {len(val_dataset)} sequences")

# ── Model ──────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prior  = ECGPrior(
    vocab_size=config.NUM_EMBEDDINGS,
    seq_len=config.SEQ_LEN,
    d_model=config.D_MODEL,
    n_heads=config.N_HEADS,
    n_layers=config.N_LAYERS,
    dropout=config.DROPOUT,
).to(device)

optimizer = torch.optim.Adam(prior.parameters(), lr=config.PRIOR_LR)
loss_fn   = nn.CrossEntropyLoss()

best_val_loss = float("inf")

# ── Training loop ──────────────────────────────────────────────────────────────
for epoch in range(1, config.PRIOR_EPOCHS + 1):

    # ── Train ──────────────────────────────────────────────────────────────────
    prior.train()
    total_train = 0.0

    for codes in train_loader:
        codes  = codes.to(device)                        # (B, T)
        logits = prior(codes)                            # (B, T, vocab_size)

        # CrossEntropyLoss wants (B, C, T) and (B, T)
        loss = loss_fn(logits.permute(0, 2, 1), codes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train += loss.item()

    # ── Validate ───────────────────────────────────────────────────────────────
    prior.eval()
    total_val = 0.0

    with torch.no_grad():
        for codes in val_loader:
            codes  = codes.to(device)
            logits = prior(codes)
            total_val += loss_fn(logits.permute(0, 2, 1), codes).item()

    avg_train = total_train / len(train_loader)
    avg_val   = total_val   / len(val_loader)

    print(f"Epoch {epoch:3d}/{config.PRIOR_EPOCHS}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}",
          end="")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(prior.state_dict(), "prior_best.pt")
        print("  * saved", end="")

    print()

print(f"\nBest val loss: {best_val_loss:.4f}  ->  prior_best.pt")
