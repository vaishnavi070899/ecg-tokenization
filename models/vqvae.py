import torch
import torch.nn as nn

from models.decoder import Decoder
from models.encoder import Encoder
from models.quantizer import ResidualVectorQuantizer, VectorQuantizer


class VQVAE(nn.Module):
    """VQ-VAE with optional Residual Vector Quantization.

    Args:
        num_rvq_stages: Number of RVQ stages.
                        1 → standard single-codebook VQ (original behaviour).
                        2+ → residual VQ; each stage quantizes the previous stage's residual.
    """

    def __init__(self, input_dim=1000, latent_dim=64, num_embeddings=512,
                 commitment_cost=0.25, decay=0.99, buffer_size=2048,
                 num_rvq_stages=1):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

        if num_rvq_stages > 1:
            self.quantizer = ResidualVectorQuantizer(
                num_stages=num_rvq_stages,
                num_embeddings=num_embeddings,
                embedding_dim=latent_dim,
                commitment_cost=commitment_cost,
                decay=decay,
                buffer_size=buffer_size,
            )
        else:
            self.quantizer = VectorQuantizer(
                num_embeddings, latent_dim, commitment_cost, decay,
                buffer_size=buffer_size,
            )

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, x):
        z_e = self.encoder(x)                                # (B, D, T)

        if isinstance(self.quantizer, ResidualVectorQuantizer):
            z_q, vq_loss, perplexity, _, residual_norms = self.quantizer(z_e)
        else:
            z_q, vq_loss, perplexity, _  = self.quantizer(z_e)
            # Quantization error in latent space — (z_e − z_q) has zero net gradient
            # (straight-through cancels), so .pow(2).mean() is safe to call here.
            residual_norms = [(z_e - z_q).pow(2).mean().item()]

        x_recon = self.decoder(z_q)                         # (B, L)
        return x_recon, vq_loss, perplexity, residual_norms

    # ── Encode / decode helpers ────────────────────────────────────────────────

    @torch.no_grad()
    def encode_indices(self, x):
        """Encode signals to discrete code index sequences.

        Used by extract_codes.py to build the dataset for prior training.

        Args:
            x: (B, L) float32 — normalised ECG signals
        Returns:
            Single-stage VQ:  (B, T)         int64
            RVQ (N stages):   (B, T, N)      int64  — stacked per-stage indices
        """
        self.eval()
        z_e     = self.encoder(x)
        out     = self.quantizer(z_e)
        indices = out[3]                              # always 4th element for both VQ and RVQ
        if isinstance(indices, list):
            return torch.stack(indices, dim=-1)       # (B, T, num_stages)
        return indices                                # (B, T)

    @torch.no_grad()
    def decode_indices(self, indices):
        """Decode code index sequences back to signals.

        Used by generate.py after sampling from the prior.

        Args:
            Single-stage VQ:  indices (B, T)       int64
            RVQ (N stages):   indices (B, T, N)    int64
        Returns:
            signals: (B, L) float32 — reconstructed ECG signals
        """
        self.eval()
        if isinstance(self.quantizer, ResidualVectorQuantizer):
            # Sum the codebook lookups from every stage
            z_q = sum(
                self.quantizer.stages[i].codebook[indices[:, :, i]]
                for i in range(self.quantizer.num_stages)
            )                                               # (B, T, D)
        else:
            z_q = self.quantizer.codebook[indices]         # (B, T, D)
        z_q = z_q.permute(0, 2, 1).contiguous()           # (B, D, T)
        return self.decoder(z_q)                           # (B, L)
