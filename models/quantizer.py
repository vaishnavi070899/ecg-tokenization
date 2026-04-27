import torch
import torch.nn as nn
# Deadcode Handling Strategy implemented: K-Means Centroid Reset (CVQ-VAE style)
# See VectorQuantizer.kmeans_centroid_reset() in models/quantizer.py for details

class VectorQuantizer(nn.Module):
    """Vector Quantization layer with Exponential Moving Average codebook updates.

    The codebook is updated via EMA rather than gradient descent, which avoids
    codebook collapse and converges faster (van den Oord et al., 2017 — appendix).

    During training, for each batch:
      N_i  ← decay * N_i  + (1 - decay) * (# vectors assigned to code i)
      m_i  ← decay * m_i  + (1 - decay) * (sum of z_e vectors assigned to code i)
      e_i   = m_i / N_i          (with Laplace smoothing to avoid /0)

    Only the commitment loss is returned — the codebook loss is replaced by EMA.

    Args:
        num_embeddings:  Codebook size K.
        embedding_dim:   Vector dimensionality D (must match latent_dim).
        commitment_cost: Weight β on the commitment loss.
        decay:           EMA decay γ (0.99 is a good default).
        epsilon:         Small value for Laplace smoothing.

    Input:  z_e  (B, D, T)  — continuous encoder output
    Output: z_q         (B, D, T)  — quantized vectors (straight-through in training)
            loss        scalar     — β * ||z_e - sg(e)||²  (commitment loss only)
            perplexity  scalar     — exp(entropy of code usage); max = num_embeddings
    """

    def __init__(self, num_embeddings=128, embedding_dim=64,
                 commitment_cost=0.25, decay=0.99, epsilon=1e-5,
                 buffer_size=2048):
        super().__init__()
        self.embedding_dim   = embedding_dim
        self.num_embeddings  = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay           = decay
        self.epsilon         = epsilon
        self.buffer_size     = buffer_size

        # Codebook: registered as a buffer so the optimizer never touches it
        codebook = torch.empty(num_embeddings, embedding_dim)
        nn.init.uniform_(codebook, -1 / num_embeddings, 1 / num_embeddings)
        self.register_buffer("codebook",          codebook)
        self.register_buffer("ema_cluster_size",  torch.zeros(num_embeddings))
        self.register_buffer("ema_embedding_sum", codebook.clone())

        # Strategy 2 (K-Means Centroid Reset) — circular buffer of recent encoder outputs
        self.register_buffer("encoder_buffer", torch.zeros(buffer_size, embedding_dim))
        self.register_buffer("buffer_ptr",     torch.tensor(0))
        self.register_buffer("buffer_full",    torch.tensor(False))

    def forward(self, z):
        z = z.permute(0, 2, 1).contiguous()      # (B, D, T) → (B, T, D)
        flat = z.view(-1, self.embedding_dim)     # (B*T, D)

        # Squared L2 distances to every codebook entry
        distances = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(1)
        )  # (B*T, K)

        indices = distances.argmin(1)                          # (B*T,)
        z_q = self.codebook[indices].view(z.shape)            # (B, T, D)
        indices_2d = indices.view(z.shape[0], z.shape[1])     # (B, T) — for prior training

        # ── Perplexity (always computed) ───────────────────────────────────────
        one_hot = torch.zeros(flat.shape[0], self.num_embeddings, device=flat.device)
        one_hot.scatter_(1, indices.unsqueeze(1), 1)          # (B*T, K)
        avg_probs  = one_hot.mean(0)                          # (K,)  usage distribution
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # ── EMA update (training only) ─────────────────────────────────────────
        if self.training:

            cluster_size  = one_hot.sum(0)                    # (K,)  reuse one_hot from above
            embedding_sum = one_hot.t() @ flat                # (K, D)

            self.ema_cluster_size.mul_(self.decay).add_(cluster_size,  alpha=1 - self.decay)
            self.ema_embedding_sum.mul_(self.decay).add_(embedding_sum, alpha=1 - self.decay)

            # Laplace smoothing so codes with 0 assignments don't collapse to 0
            n = self.ema_cluster_size.sum()
            smoothed = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )                                                  # (K,)
            self.codebook.copy_(self.ema_embedding_sum / smoothed.unsqueeze(1))

            # ── Codebook Reset — call site (pick ONE strategy, keep others commented) ─
            # Step 1: detect dead codes (shared by all three strategies)
            #
            dead_indices, dead_mask = self.find_dead_codes(
                ema_cluster_size = self.ema_cluster_size,
                threshold        = 1.0,
            )  # dead_indices: 1-D LongTensor of dead code positions; dead_mask: boolean (K,) tensor
            #
            # ── Strategy 1: Random Restart ────────────────────────────────────
            # self.random_restart(
            #     dead_indices = dead_indices,
            #     flat         = flat,
            # )
            #
            # ── Strategy 2: K-Means Centroid Reset ───────────────────────────
            self.update_buffer(flat)                   # keep buffer current
            self.kmeans_centroid_reset(
                dead_indices = dead_indices,
            )
            #
            # ── Strategy 3: Anchor Resampling ────────────────────────────────
            # active_codes = (~dead_mask).nonzero(as_tuple=True)[0]  # live code indices
            # self.anchor_resampling(
            #     dead_indices = dead_indices,
            #     flat         = flat,
            #     active_codes = active_codes,
            # )

        # Commitment loss: push encoder outputs toward codebook (sg on codebook)
        commitment_loss = self.commitment_cost * (z - z_q.detach()).pow(2).mean()

        # Straight-through estimator
        z_q = z + (z_q - z).detach()                         # (B, T, D)
        z_q = z_q.permute(0, 2, 1).contiguous()              # (B, D, T)

        return z_q, commitment_loss, perplexity, indices_2d
    
    # ── Strategy 3 method stubs (all commented out — implement before enabling) ──
    #
    def find_dead_codes(self, ema_cluster_size, threshold=1.0):             #Threshold is set at 1.0, meaning codes that have been assigned less than once on average across recent batches are considered dead.
        """Return indices and mask of dead codebook entries.
    
        A code is dead if its EMA-smoothed assignment count has fallen
        below `threshold` (i.e. it has been assigned less than once on
        average across recent batches).
    
        Args:
            ema_cluster_size: (K,) tensor — self.ema_cluster_size
            threshold:        float — codes below this count are dead       #set at 1.0
    
        Returns:
            dead_indices: 1-D LongTensor of dead code positions
            dead_mask:    boolean (K,) tensor
        """
        dead_mask    = (ema_cluster_size < threshold)          # boolean (K,)
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]     # 1-D LongTensor
        return dead_indices, dead_mask

    def random_restart(self, dead_indices, flat):
        """Strategy 1 — replace each dead code with a randomly chosen encoder
        output from the current batch.
    
        Args:
            dead_indices: 1-D LongTensor — positions of dead codes in codebook
            flat:         (B*T, D) tensor — flattened encoder outputs
    
        Side-effects (all in-place):
            self.codebook[dead_idx]          <- sampled encoder vector
            self.ema_cluster_size[dead_idx]  <- 1.0
            self.ema_embedding_sum[dead_idx] <- sampled encoder vector
        """
        if len(dead_indices) == 0:
            return                                             # nothing to do
    
        n_dead = len(dead_indices)
        n_flat = flat.shape[0]                               # B*T available vectors
    
        # Sample n_dead random positions from the current batch (with replacement)
        sampled_positions  = torch.randint(0, n_flat, (n_dead,), device=flat.device)
        replacement_vectors = flat[sampled_positions]        # (n_dead, D)
    
        # Overwrite dead code slots and reset EMA accumulators
        for i, dead_idx in enumerate(dead_indices):
            self.codebook[dead_idx]          = replacement_vectors[i].detach()
            self.ema_cluster_size[dead_idx]  = 1.0
            self.ema_embedding_sum[dead_idx] = replacement_vectors[i].detach()

    # ── Strategy 2 helper — must be called every forward pass before reset ──────
    
    def update_buffer(self, flat):
        """Append current batch's encoder outputs to the circular buffer.
    
        New state required in __init__:
            BUFFER_SIZE = 2048   (add to config.py when ready)
            self.register_buffer("encoder_buffer",
                                 torch.zeros(BUFFER_SIZE, self.embedding_dim))
            self.register_buffer("buffer_ptr",   torch.tensor(0))
            self.register_buffer("buffer_full",  torch.tensor(False))
        """
        n        = flat.shape[0]
        buf_size = self.encoder_buffer.shape[0]
        ptr      = int(self.buffer_ptr)

        if n >= buf_size:
            # Batch is larger than the buffer — overwrite everything with the
            # most recent buf_size vectors and reset the pointer to 0
            self.encoder_buffer.copy_(flat[-buf_size:].detach())
            self.buffer_ptr  = torch.tensor(0)
            self.buffer_full = torch.tensor(True)
        elif ptr + n <= buf_size:
            # Fits without wrapping
            self.encoder_buffer[ptr:ptr + n] = flat.detach()
            self.buffer_ptr = torch.tensor((ptr + n) % buf_size)
        else:
            # Wrap around the circular buffer
            split = buf_size - ptr
            self.encoder_buffer[ptr:]       = flat[:split].detach()
            self.encoder_buffer[:n - split] = flat[split:].detach()
            self.buffer_ptr  = torch.tensor(n - split)
            self.buffer_full = torch.tensor(True)

    def kmeans_centroid_reset(self, dead_indices, n_iters=10):
        """Strategy 2 — reset dead codes to mini K-means centroids computed
        over recently seen encoder outputs (CVQ-VAE style).
    
        Args:
            dead_indices: 1-D LongTensor — positions of dead codes in codebook
            n_iters:      int — number of mini K-means iterations
    
        Requires update_buffer() to have been called this forward pass.
    
        Side-effects (all in-place):
            self.codebook[dead_idx]          <- centroid vector
            self.ema_cluster_size[dead_idx]  <- 1.0
            self.ema_embedding_sum[dead_idx] <- centroid vector
        """
        if len(dead_indices) == 0:
            return                                             # nothing to do
    
        # Use however much of the buffer has been filled so far
        buf_size = self.encoder_buffer.shape[0]
        ptr      = int(self.buffer_ptr)
        active_buffer = (self.encoder_buffer
                         if self.buffer_full
                         else self.encoder_buffer[:ptr])      # (N_buf, D)
    
        if active_buffer.shape[0] < len(dead_indices):
            return                                             # not enough data yet
    
        # Randomly initialise one centroid per dead code from the buffer
        init_pos  = torch.randperm(active_buffer.shape[0])[:len(dead_indices)]
        centroids = active_buffer[init_pos].clone()           # (n_dead, D)
    
        # Mini K-means: iterate assignment → centroid recompute
        for _ in range(n_iters):
            # Squared-L2 distances: (N_buf, n_dead)
            dists = (
                active_buffer.pow(2).sum(1, keepdim=True)
                - 2 * active_buffer @ centroids.t()
                + centroids.pow(2).sum(1)
            )
            assignments = dists.argmin(dim=1)                 # (N_buf,)
    
            # Recompute each centroid as the mean of its assigned vectors
            for j in range(len(dead_indices)):
                members = active_buffer[assignments == j]
                if members.shape[0] > 0:
                    centroids[j] = members.mean(dim=0)
                # else: keep previous centroid unchanged
    
        # Write centroids into dead code slots and reset EMA accumulators
        for i, dead_idx in enumerate(dead_indices):
            self.codebook[dead_idx]          = centroids[i].detach()
            self.ema_cluster_size[dead_idx]  = 1.0
            self.ema_embedding_sum[dead_idx] = centroids[i].detach()

    def anchor_resampling(self, dead_indices, flat, active_codes):
        """Reset dead codes to encoder outputs that are least covered by
        the currently active codebook entries (importance-weighted sampling).
    
        Args:
            dead_indices: 1-D LongTensor — positions of dead codes in codebook
            flat:         (B*T, D) tensor — flattened encoder outputs (in scope
                          inside forward() after the EMA update block)
            active_codes: 1-D LongTensor — indices of live codes
                          (~dead_mask).nonzero(as_tuple=True)[0]
    
        Side-effects (all in-place):
            self.codebook[dead_idx]          <- anchor vector
            self.ema_cluster_size[dead_idx]  <- 1.0
            self.ema_embedding_sum[dead_idx] <- anchor vector
        """
        if len(dead_indices) == 0:
            return                                             # nothing to do
    
        # ── Step 1: gather active codebook entries ─────────────────────────
        active_codebook = self.codebook[active_codes]         # (n_active, D)
    
        # ── Step 2: distance from every encoder output to its nearest active code
        #    Reuse the same squared-L2 trick used in forward():
        #      ||a - b||^2 = ||a||^2 - 2 a·b^T + ||b||^2
        distances = (
            flat.pow(2).sum(1, keepdim=True)                  # (B*T, 1)
            - 2 * flat @ active_codebook.t()                  # (B*T, n_active)
            + active_codebook.pow(2).sum(1)                   # (n_active,)
        )                                                      # (B*T, n_active)
        min_distance = distances.min(dim=1).values            # (B*T,)  — dist to nearest active code
        min_distance = min_distance.clamp(min=0)              # numerical safety (no negatives from fp)
    
        # ── Step 3: convert distances to sampling weights ──────────────────
        #    Encoder outputs that are far from all active codes get higher weight.
        #    Add small epsilon so zero-distance outputs still have a nonzero weight.
        weights = min_distance + 1e-6                         # (B*T,)
        weights = weights / weights.sum()                     # normalise -> probability distribution
    
        # ── Step 4: sample one anchor per dead code (with replacement) ─────
        sampled_positions = torch.multinomial(
            weights,
            num_samples=len(dead_indices),
            replacement=True,
        )                                                      # (n_dead,)
        anchors = flat[sampled_positions]                     # (n_dead, D)
    
        # ── Step 5: overwrite dead code slots and reset EMA accumulators ───
        for i, dead_idx in enumerate(dead_indices):
            self.codebook[dead_idx]          = anchors[i].detach()
            self.ema_cluster_size[dead_idx]  = 1.0
            self.ema_embedding_sum[dead_idx] = anchors[i].detach()


# ── Residual Vector Quantizer ──────────────────────────────────────────────────

class ResidualVectorQuantizer(nn.Module):
    """Two-stage Residual VQ layer.

    Stage 1 quantizes the encoder output z_e normally.
    Stage 2 quantizes the residual  r = z_e − z_q1.
    The decoder receives z_q1 + z_q2.

    Each stage is an independent VectorQuantizer with its own codebook and EMA
    accumulators, so the two codebooks specialise on different signal components.

    Straight-through gradients flow back through both stages independently:
      ∂(z_q1)/∂z_e = 1   (stage-1 straight-through)
      ∂(z_q2)/∂z_e = 1   (stage-2 straight-through via residual = z_e − z_q1.detach())
    The encoder therefore receives gradient signal from both stages, which
    strengthens the learning signal relative to single-stage VQ.

    Args:
        num_stages:      Number of RVQ stages (default 2).
        num_embeddings:  Codebook size K (shared across stages).
        embedding_dim:   Vector dimensionality D.
        commitment_cost: β — weight on commitment loss (shared across stages).
        decay:           EMA decay γ (shared across stages).
        epsilon:         Laplace smoothing constant.
        buffer_size:     Circular buffer size for K-Means Centroid Reset.

    Input:  z_e  (B, D, T)  — continuous encoder output
    Output: z_q         (B, D, T)  — sum of quantized outputs from all stages
            vq_loss     scalar     — sum of commitment losses across stages
            perplexity  scalar     — average perplexity across stages
            all_indices list[(B, T)] — per-stage codebook index tensors
    """

    def __init__(self, num_stages=2, num_embeddings=512, embedding_dim=64,
                 commitment_cost=0.25, decay=0.99, epsilon=1e-5, buffer_size=2048):
        super().__init__()
        self.num_stages = num_stages
        self.stages = nn.ModuleList([
            VectorQuantizer(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
                decay=decay,
                epsilon=epsilon,
                buffer_size=buffer_size,
            )
            for _ in range(num_stages)
        ])

    def forward(self, z):
        """
        Args:
            z: (B, D, T) — encoder output

        Returns:
            z_q_total:     (B, D, T)
            total_vq_loss: scalar
            avg_perplexity: scalar
            all_indices:   list of (B, T) int tensors, length = num_stages
        """
        z_q_total      = torch.zeros_like(z)
        total_vq_loss  = z.new_tensor(0.0)
        perplexities   = []
        all_indices    = []
        residual_norms = []
        residual       = z

        for stage in self.stages:
            z_q_stage, loss, perplexity, indices = stage(residual)

            z_q_total     = z_q_total + z_q_stage
            total_vq_loss = total_vq_loss + loss
            perplexities.append(perplexity)
            all_indices.append(indices)

            # Residual for the next stage: subtract the hard-quantized value.
            # z_q_stage uses straight-through, so z_q_stage.detach() == z_q_hard.detach().
            residual = residual - z_q_stage.detach()

            # Track residual MSE after this stage (no gradient needed)
            residual_norms.append(residual.pow(2).mean().item())

        avg_perplexity = sum(perplexities) / self.num_stages
        return z_q_total, total_vq_loss, avg_perplexity, all_indices, residual_norms
