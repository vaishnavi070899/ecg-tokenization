import torch.nn as nn


class Encoder(nn.Module):
    """1-D convolutional encoder for ECG signals.

    Downsamples the input by 8x via three stride-2 convolutions.

    Args:
        input_dim:  Length of the input signal (unused by Conv1d, kept for API compatibility).
        latent_dim: Number of channels in the output latent representation.

    Input:  (B, L)  or  (B, 1, L)
    Output: (B, latent_dim, L // 8)
    """

    def __init__(self, input_dim=1000, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1),       # → (B, 32,  L/2)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),      # → (B, 64,  L/4)
            nn.ReLU(),
            nn.Conv1d(64, latent_dim, kernel_size=4, stride=2, padding=1),  # → (B, D,  L/8)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (B, L) → (B, 1, L)
        return self.conv(x)             # (B, latent_dim, L//8)
