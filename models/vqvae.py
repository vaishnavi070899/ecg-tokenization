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
                 num_rvq_stages=1, use_nsvq=False, nsvq_tau_per_stage=None,
                 nsvq_alpha=0.1, nsvq_repulse_gamma=0.05, nsvq_at_risk_thresh=5.0):
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
                use_nsvq=use_nsvq,
                tau_per_stage=nsvq_tau_per_stage,
                nsvq_alpha=nsvq_alpha,
                nsvq_repulse_gamma=nsvq_repulse_gamma,
                nsvq_at_risk_thresh=nsvq_at_risk_thresh,
            )
        else:
            tau = nsvq_tau_per_stage[0] if nsvq_tau_per_stage else 1.0
            self.quantizer = VectorQuantizer(
                num_embeddings, latent_dim, commitment_cost, decay,
                buffer_size=buffer_size,
                use_nsvq=use_nsvq,
                tau=tau,
                nsvq_alpha=nsvq_alpha,
                nsvq_repulse_gamma=nsvq_repulse_gamma,
                nsvq_at_risk_thresh=nsvq_at_risk_thresh,
            )

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, x):
        z_e = self.encoder(x)                                # (B, D, T)

        if isinstance(self.quantizer, ResidualVectorQuantizer):
            z_q, vq_loss, perplexity, _, residual_norms = self.quantizer(z_e)
        else:
            z_q, vq_loss, perplexity, _ = self.quantizer(z_e)
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
    def warm_start_codebooks(self, data_loader, device, n_batches=10):
        """Initialize each RVQ stage's codebook from its actual residual distribution.

        The default uniform_(-1/K, 1/K) init puts all codebooks at vector norm
        ~0.02, whereas real residuals at Stage 1 have norm ~1.6, Stage 2 ~1.1,
        Stage 3 ~0.8, Stage 4 ~0.6.  Starting 30-80x too small means the first
        several batches are wasted re-scaling every code to the right range
        before any useful spread occurs.

        This method collects residuals from the first n_batches of real data,
        then uses k-means++ to seed each stage with K diverse initial centroids
        drawn from its own residual distribution.

        Only applies when quantizer is a ResidualVectorQuantizer (no-op for
        single-stage VQ which already converges quickly).

        Args:
            data_loader: training DataLoader
            device:      torch.device
            n_batches:   how many batches to collect residuals from (default 10)
        """
        if not isinstance(self.quantizer, ResidualVectorQuantizer):
            return

        self.eval()
        D          = self.quantizer.stages[0].embedding_dim
        num_stages = self.quantizer.num_stages

        # Sequential warm-start: initialise each stage from residuals produced
        # by the ALREADY-INITIALISED earlier stages, not the near-zero defaults.
        # This means Stage 2 sees true Stage-1 residuals, Stage 3 sees true
        # Stage-2 residuals, etc.
        for s, stage in enumerate(self.quantizer.stages):
            pool = []

            for i, x in enumerate(data_loader):
                if i >= n_batches:
                    break
                x    = x.to(device)
                z_e  = self.encoder(x)                         # (B, D, T)
                flat = z_e.permute(0, 2, 1).contiguous().view(-1, D)  # (B*T, D)

                residual = flat.clone()
                # Step through stages 0 … s-1 (already warm-started)
                for prev_stage in self.quantizer.stages[:s]:
                    dist = (
                        residual.pow(2).sum(1, keepdim=True)
                        - 2 * residual @ prev_stage.codebook.t()
                        + prev_stage.codebook.pow(2).sum(1)
                    ).clamp(min=0)
                    residual = residual - prev_stage.codebook[dist.argmin(1)]

                pool.append(residual.cpu())

            pool_t = torch.cat(pool, dim=0)                    # (N, D)
            K      = stage.num_embeddings
            init   = self._kmeans_plus_plus(pool_t, K).to(device)  # (K, D)

            stage.codebook.copy_(init)
            stage.ema_embedding_sum.copy_(init)
            stage.ema_cluster_size.fill_(1.0)                  # warm EMA — no instant reset

            norms = pool_t.norm(dim=1)
            print(f"  Warm-start Stage {s+1}: "
                  f"pool={pool_t.shape[0]}  "
                  f"residual norm  mean={norms.mean():.4f}  std={norms.std():.4f}  "
                  f"codebook norm  mean={init.norm(dim=1).mean().item():.4f}")

        self.train()

    @staticmethod
    def _kmeans_plus_plus(pool, K):
        """Select K diverse initial centroids from pool using k-means++ seeding.

        Much better initial coverage than uniform random sampling:
        subsequent centroids are sampled with probability proportional to
        their squared distance from the nearest existing centroid, so they
        naturally spread across the distribution.

        Args:
            pool: (N, D) tensor of candidate vectors (on CPU)
            K:    number of centroids to pick

        Returns:
            (K, D) tensor of initial centroids
        """
        N = pool.shape[0]
        # First centroid: random
        idx      = torch.randint(N, (1,)).item()
        centers  = [pool[idx]]

        for _ in range(K - 1):
            stacked = torch.stack(centers, dim=0)              # (k, D)
            # Squared L2 distance from each pool point to its nearest center
            sq_dist = (
                pool.pow(2).sum(1, keepdim=True)               # (N, 1)
                - 2 * pool @ stacked.t()                       # (N, k)
                + stacked.pow(2).sum(1)                        # (k,)
            ).clamp(min=0).min(dim=1).values                   # (N,)

            probs      = sq_dist / sq_dist.sum().clamp(min=1e-10)
            next_idx   = torch.multinomial(probs, 1).item()
            centers.append(pool[next_idx])

        return torch.stack(centers, dim=0)                     # (K, D)

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
