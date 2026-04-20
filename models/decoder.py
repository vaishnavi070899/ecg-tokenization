import torch.nn as nn


class Decoder(nn.Module):
    """1-D transposed-convolutional decoder for ECG signals.

    Upsamples the latent representation by 8x, mirroring the Encoder.

    Args:
        latent_dim: Number of channels in the input latent representation.
        output_dim: Length of the reconstructed signal (unused by ConvTranspose1d,
                    kept for API compatibility).

    Input:  (B, latent_dim, T)
    Output: (B, T * 8)
    """

    def __init__(self, latent_dim=64, output_dim=1000):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 64, kernel_size=4, stride=2, padding=1),  # → (B, 64, T*2)
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),          # → (B, 32, T*4)
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),           # → (B,  1, T*8)
        )

    def forward(self, z):
        out = self.deconv(z)    # (B, 1, T*8)
        return out.squeeze(1)   # (B, T*8)
