"""sweep_embeddings.py

Evaluate the trained VQ-VAE checkpoint and log:
  - Mean reconstruction MSE
  - Active codes / dead codes (%)
  - Code frequency histogram

Outputs
-------
  recon.png      — original vs reconstructed signals
  codebook.png   — sorted frequency plot + histogram of frequencies
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from data.load_data import PTBXLDataset
from models.vqvae import VQVAE

# ── Config ─────────────────────────────────────────────────────────────────────
N_RECORDS = 8    # val records used for every metric
N_DISPLAY = 4    # records shown in the recon plot


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_model(device):
    ckpt  = "vqvae_best.pt"
    model = VQVAE(input_dim=config.INPUT_DIM, latent_dim=config.LATENT_DIM,
                  num_embeddings=config.NUM_EMBEDDINGS)
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        trained = True
        print(f"Loaded {ckpt}")
    else:
        trained = False
        print(f"WARNING: {ckpt} not found -- using random weights")
    return model.to(device).eval(), trained


def compute_mse(model, signals, device):
    with torch.no_grad():
        recons, _, _ = model(signals.to(device))
    return signals.numpy(), recons.cpu().numpy()


def compute_codebook_usage(model, signals, device, num_embeddings):
    """Return usage stats for every codebook entry across all signals.

    Returns a dict with:
      counts      (K,) — how many times each code was assigned
      used_codes  int  — number of codes assigned at least once
      dead_codes  int  — codes never assigned
      dead_pct    float
      perplexity  float — exp(entropy of usage distribution)
      total       int  — total code assignments = N * seq_len
    """
    with torch.no_grad():
        indices = model.encode_indices(signals.to(device))   # (N, 125)
    flat   = indices.cpu().numpy().flatten()                  # (N*125,)
    counts = np.bincount(flat, minlength=num_embeddings)      # (K,)

    total      = len(flat)
    used_codes = int((counts > 0).sum())
    dead_codes = num_embeddings - used_codes
    dead_pct   = dead_codes / num_embeddings * 100

    probs      = counts / total
    entropy    = -np.sum(probs * np.log(probs + 1e-10))
    perplexity = float(np.exp(entropy))

    return dict(counts=counts, used_codes=used_codes, dead_codes=dead_codes,
                dead_pct=dead_pct, perplexity=perplexity, total=total)


# ── Plots ──────────────────────────────────────────────────────────────────────
def save_recon_plot(originals, recons, mse_per_signal, K, trained, path):
    n    = min(N_DISPLAY, len(originals))
    fig, axes = plt.subplots(n, 1, figsize=(12, n * 2.2))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(originals[i], color="steelblue", linewidth=0.9, label="Original")
        ax.plot(recons[i],    color="tomato",    linewidth=0.9, label="Reconstructed", alpha=0.85)
        ax.set_title(f"Record {i+1}   MSE={mse_per_signal[i]:.4f}", fontsize=9)
        ax.set_xlabel("Time step (100 Hz)", fontsize=7)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
    status = f"K={K}" + ("" if trained else "  [UNTRAINED]")
    fig.suptitle(f"VQ-VAE Reconstruction   {status}   mean MSE={mse_per_signal.mean():.4f}",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def save_codebook_histogram(usage, K, trained, path):
    """Two-panel figure:
       Left  — sorted code frequencies (descending), dead codes in red
       Right — histogram of frequency values (how many codes appear N times)
    """
    counts = usage["counts"]
    sorted_counts = np.sort(counts)[::-1]          # descending

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # ── Left: sorted frequency plot ───────────────────────────────────────────
    colors = ["#4c8cbf" if c > 0 else "#e05c5c" for c in sorted_counts]
    axes[0].bar(np.arange(K), sorted_counts, color=colors, width=1.0, linewidth=0)
    axes[0].set_xlabel("Code rank (sorted by frequency)", fontsize=9)
    axes[0].set_ylabel("Assignment count", fontsize=9)
    axes[0].set_title(f"Code Usage Frequencies   K={K}", fontsize=10)

    # Annotations
    info = (f"Used:       {usage['used_codes']}/{K}\n"
            f"Dead:       {usage['dead_codes']}  ({usage['dead_pct']:.1f}%)\n"
            f"Perplexity: {usage['perplexity']:.1f}")
    axes[0].text(0.98, 0.97, info, transform=axes[0].transAxes,
                 fontsize=8, va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

    used_patch = plt.Rectangle((0, 0), 1, 1, fc="#4c8cbf", label="Used")
    dead_patch = plt.Rectangle((0, 0), 1, 1, fc="#e05c5c", label="Dead (0 assignments)")
    axes[0].legend(handles=[used_patch, dead_patch], fontsize=8, loc="upper right",
                   bbox_to_anchor=(0.98, 0.70))

    # ── Right: histogram of frequency values ──────────────────────────────────
    nonzero = counts[counts > 0]
    if len(nonzero) > 0:
        axes[1].hist(nonzero, bins=min(40, len(nonzero)),
                     color="#4c8cbf", edgecolor="white", linewidth=0.4)
    axes[1].set_xlabel("Assignment count per code", fontsize=9)
    axes[1].set_ylabel("Number of codes", fontsize=9)
    axes[1].set_title(f"Frequency Distribution (used codes only)   K={K}", fontsize=10)
    axes[1].axvline(nonzero.mean() if len(nonzero) else 0,
                    color="red", linestyle="--", linewidth=1.2,
                    label=f"mean={nonzero.mean():.1f}" if len(nonzero) else "")
    if len(nonzero):
        axes[1].legend(fontsize=8)

    status = "" if trained else "  [UNTRAINED]"
    fig.suptitle(f"Codebook Usage Analysis   K={K}{status}", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = PTBXLDataset(folds=[10], n_records=N_RECORDS)
signals = torch.stack([dataset[i] for i in range(N_RECORDS)])

print(f"\nDevice: {device}   |   Val records: {N_RECORDS}   |   K={config.NUM_EMBEDDINGS}\n")

model, trained = load_model(device)
K = config.NUM_EMBEDDINGS

# Reconstruction MSE
originals, recons = compute_mse(model, signals, device)
mse_per_signal    = ((originals - recons) ** 2).mean(axis=1)

# Codebook usage
usage = compute_codebook_usage(model, signals, device, K)

# Console log
print(f"Mean MSE     : {mse_per_signal.mean():.4f}")
print(f"Active codes : {usage['used_codes']}/{K}")
print(f"Dead codes   : {usage['dead_codes']}  ({usage['dead_pct']:.1f}%)")
print()

# Plots
save_recon_plot(originals, recons, mse_per_signal, K=K, trained=trained, path="recon.png")
save_codebook_histogram(usage, K=K, trained=trained, path="codebook.png")