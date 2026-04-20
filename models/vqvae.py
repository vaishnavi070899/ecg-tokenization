import torch
import torch.nn as nn

from models.decoder import Decoder
from models.encoder import Encoder
from models.quantizer import VectorQuantizer


class VQVAE(nn.Module):
    def __init__(self, input_dim=1000, latent_dim=64, num_embeddings=512,
                 commitment_cost=0.25, decay=0.99, buffer_size=2048):
        super().__init__()
        self.encoder   = Encoder(input_dim, latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost, decay,
                                         buffer_size=buffer_size)
        self.decoder   = Decoder(latent_dim, input_dim)

    def forward(self, x):
        z_e                              = self.encoder(x)       # (B, D, T)
        z_q, vq_loss, perplexity, _      = self.quantizer(z_e)   # (B, D, T), scalar, scalar, (B, T)
        x_recon                          = self.decoder(z_q)     # (B, L)
        return x_recon, vq_loss, perplexity

    @torch.no_grad()
    def encode_indices(self, x):
        """Encode a batch of signals to discrete code index sequences.

        Used by extract_codes.py to build the dataset for prior training.

        Args:
            x: (B, L) float32 — normalised ECG signals
        Returns:
            indices: (B, T) int64 — codebook indices in [0, num_embeddings)
        """
        self.eval()
        z_e                          = self.encoder(x)
        _, _, _, indices             = self.quantizer(z_e)
        return indices

    @torch.no_grad()
    def decode_indices(self, indices):
        """Decode a batch of code index sequences back to signals.

        Used by generate.py after sampling from the prior.

        Args:
            indices: (B, T) int64 — codebook indices
        Returns:
            signals: (B, L) float32 — reconstructed ECG signals
        """
        self.eval()
        z_q = self.quantizer.codebook[indices]          # (B, T, D)
        z_q = z_q.permute(0, 2, 1).contiguous()        # (B, D, T)
        return self.decoder(z_q)                        # (B, L)
