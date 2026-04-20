"""reconstruct.py

Load real ECG records, pass them through the VQ-VAE, and plot
original vs reconstructed side-by-side to assess morphology.

Usage
-----
    python reconstruct.py                        # uses vqvae_best.pt, 4 records
    python reconstruct.py --n 8                  # show 8 records
    python reconstruct.py --checkpoint my.pt     # custom checkpoint
    python reconstruct.py --save recon.png       # save instead of interactive plot
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from data.load_data import PTBXLDataset
from models.vqvae import VQVAE


def load_model(checkpoint, device):
    model = VQVAE(input_dim=config.INPUT_DIM, latent_dim=config.LATENT_DIM,
                  num_embeddings=config.NUM_EMBEDDINGS)
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print(f"WARNING: '{checkpoint}' not found -- using random weights.")
        print("         Run main.py first to get a trained checkpoint.")
    return model.to(device).eval()


def reconstruct(n=4, checkpoint="vqvae_best.pt", save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(checkpoint, device)

    # Load n real records from the val fold so they weren't seen during training
    dataset = PTBXLDataset(folds=[10], n_records=n)
    signals = torch.stack([dataset[i] for i in range(n)]).to(device)  # (n, 1000)

    with torch.no_grad():
        recons, _, _ = model(signals)   # (n, 1000)

    originals = signals.cpu().numpy()
    recons    = recons.cpu().numpy()

    # ── Compute per-signal MSE ─────────────────────────────────────────────────
    mse_per_signal = ((originals - recons) ** 2).mean(axis=1)

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n, 1, figsize=(12, n * 2.2))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(originals[i], color="steelblue", linewidth=0.9, label="Original")
        ax.plot(recons[i],    color="tomato",    linewidth=0.9, label="Reconstructed", alpha=0.85)
        ax.set_title(f"Record {i + 1}   MSE = {mse_per_signal[i]:.4f}", fontsize=9)
        ax.set_xlabel("Time step (100 Hz)", fontsize=7)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    trained = os.path.exists(checkpoint)
    status  = checkpoint if trained else "UNTRAINED (random weights)"
    fig.suptitle(f"VQ-VAE Reconstruction Check   [{status}]", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    # ── Console summary ────────────────────────────────────────────────────────
    print(f"\nMSE per signal: {np.round(mse_per_signal, 4)}")
    print(f"Mean MSE:       {mse_per_signal.mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default="vqvae_best.pt")
    parser.add_argument("--save",       type=str, default=None)
    args = parser.parse_args()

    reconstruct(n=args.n, checkpoint=args.checkpoint, save_path=args.save)
