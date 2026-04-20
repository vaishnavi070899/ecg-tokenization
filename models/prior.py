import torch
import torch.nn as nn


class ECGPrior(nn.Module):
    """Autoregressive Transformer prior over VQ-VAE code sequences.

    Trained on sequences of discrete codebook indices produced by the frozen
    VQ-VAE encoder+quantizer. Models P(c_t | c_1..c_{t-1}) via causal
    (GPT-style) self-attention.

    Args:
        vocab_size: Codebook size (must match VQ-VAE NUM_EMBEDDINGS).
        seq_len:    Code sequence length (input_length // 8 = 125 for 1000-sample ECG).
        d_model:    Transformer embedding dimension.
        n_heads:    Number of attention heads (d_model must be divisible by n_heads).
        n_layers:   Number of Transformer layers.
        dropout:    Dropout probability.
    """

    def __init__(self, vocab_size=512, seq_len=125, d_model=128,
                 n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        self.vocab_size  = vocab_size
        self.seq_len     = seq_len
        self.start_token = vocab_size  # index 512 — distinct from all codebook entries

        # +1 to accommodate the start token at index vocab_size
        self.token_emb = nn.Embedding(vocab_size + 1, d_model)
        self.pos_emb   = nn.Embedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

        # Pre-built causal mask for full-length sequences; moves to GPU with the model
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(seq_len)
        )

    def forward(self, codes):
        """Teacher-forced forward pass used during training.

        Args:
            codes: (B, seq_len) int64 in [0, vocab_size)
        Returns:
            logits: (B, seq_len, vocab_size)
        """
        B, T = codes.shape

        # Shift right: prepend start token, drop the last code.
        # Input  → [<start>, c_0,   c_1,   ..., c_{T-2}]
        # Target → [c_0,     c_1,   c_2,   ..., c_{T-1}]   (= codes, unshifted)
        start = torch.full((B, 1), self.start_token, dtype=torch.long, device=codes.device)
        x = torch.cat([start, codes[:, :-1]], dim=1)               # (B, T)

        positions = torch.arange(T, device=codes.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(positions)            # (B, T, d_model)

        x = self.transformer(x, mask=self.causal_mask)             # (B, T, d_model)
        return self.head(x)                                         # (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, n_samples=1, temperature=1.0, device="cpu"):
        """Autoregressively sample code sequences.

        Args:
            n_samples:   Number of sequences to generate in parallel.
            temperature: Softmax temperature (< 1 = sharper, > 1 = more random).
            device:      Torch device string or object.

        Returns:
            codes: (n_samples, seq_len) int64 tensor of codebook indices.
        """
        self.eval()
        tokens = torch.full((n_samples, 1), self.start_token, dtype=torch.long, device=device)

        for _ in range(self.seq_len):
            T_cur     = tokens.shape[1]
            positions = torch.arange(T_cur, device=device).unsqueeze(0)
            x         = self.token_emb(tokens) + self.pos_emb(positions)  # (B, T_cur, d_model)

            mask = nn.Transformer.generate_square_subsequent_mask(T_cur, device=device)
            x    = self.transformer(x, mask=mask)                          # (B, T_cur, d_model)

            logits     = self.head(x[:, -1, :]) / temperature             # (B, vocab_size)
            probs      = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)          # (B, 1)
            tokens     = torch.cat([tokens, next_token], dim=1)

        return tokens[:, 1:]   # strip start token → (n_samples, seq_len)
