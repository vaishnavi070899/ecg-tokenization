"""generate.py

Sample synthetic ECG signals using the trained VQ-VAE prior.

Pipeline
--------
1. Load frozen VQ-VAE  (vqvae_best.pt)
2. Load trained prior  (prior_best.pt)
3. Sample N code sequences from the prior autoregressively
4. Decode each code sequence -> ECG signal via VQ-VAE decoder
5. Plot and save the results

Usage
-----
    python generate.py                  # generate 8 signals, default temperature
    python generate.py --n 16           # generate 16 signals
    python generate.py --temp 0.8       # sharper (less random) sampling
    python generate.py --save ecgs.png  # save plot to file instead of showing
"""

import argparse

import torch

import config
from models.prior import ECGPrior
from models.vqvae import VQVAE
from utils.plot import plot_ecg_grid


def load_vqvae(checkpoint, device):
    model = VQVAE(input_dim=config.INPUT_DIM, latent_dim=config.LATENT_DIM,
                  num_embeddings=config.NUM_EMBEDDINGS)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()


def load_prior(checkpoint, device):
    prior = ECGPrior(vocab_size=config.NUM_EMBEDDINGS, seq_len=config.SEQ_LEN,
                     d_model=config.D_MODEL, n_heads=config.N_HEADS, n_layers=config.N_LAYERS)
    prior.load_state_dict(torch.load(checkpoint, map_location=device))
    return prior.to(device).eval()


def generate(n_samples=8, temperature=1.0,
             vqvae_ckpt="vqvae_best.pt", prior_ckpt="prior_best.pt",
             save_path=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading VQ-VAE from {vqvae_ckpt} ...")
    vqvae = load_vqvae(vqvae_ckpt, device)

    print(f"Loading prior from  {prior_ckpt} ...")
    prior = load_prior(prior_ckpt, device)

    # ── Sample code sequences from the prior ───────────────────────────────────
    print(f"Sampling {n_samples} code sequences (temperature={temperature}) ...")
    codes = prior.generate(n_samples=n_samples, temperature=temperature, device=device)

    # ── Decode to ECG signals ──────────────────────────────────────────────────
    signals = vqvae.decode_indices(codes).cpu().numpy()   # (n_samples, 1000)
    print(f"Generated signals: {signals.shape}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    plot_ecg_grid(
        signals,
        title=f"{n_samples} Synthetic ECGs  (temp={temperature})",
        save_path=save_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int,   default=8,    help="Number of signals to generate")
    parser.add_argument("--temp",  type=float, default=1.0,  help="Sampling temperature")
    parser.add_argument("--vqvae", type=str,   default="vqvae_best.pt")
    parser.add_argument("--prior", type=str,   default="prior_best.pt")
    parser.add_argument("--save",  type=str,   default=None, help="Path to save plot (e.g. ecgs.png)")
    args = parser.parse_args()

    generate(
        n_samples=args.n,
        temperature=args.temp,
        vqvae_ckpt=args.vqvae,
        prior_ckpt=args.prior,
        save_path=args.save,
    )
