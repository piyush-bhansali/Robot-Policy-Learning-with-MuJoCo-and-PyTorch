"""Convolutional VAE for 96 × 96 RGB Reacher images."""
import torch
from torch import nn
from torch.nn import functional as F


def vae_loss(
    reconstruction: torch.Tensor,
    images: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    
    if not 0 <= beta < float("inf"):
        raise ValueError("beta must be finite and nonnegative")
    if reconstruction.shape != images.shape:
        raise ValueError("reconstruction and images must have matching shapes")
    if mu.ndim != 2 or mu.shape != logvar.shape or mu.shape[0] != images.shape[0]:
        raise ValueError("mu and logvar must have matching (batch, latent_dim) shapes")

    reconstruction_loss = F.mse_loss(reconstruction, images, reduction="none")
    reconstruction_loss = reconstruction_loss.flatten(1).sum(dim=1).mean()

    # KL(N(mu, diag(exp(logvar))) || N(0, I)), calculated per image.
    kl_loss = -0.5 * (1 + logvar - mu.square() - logvar.exp())
    kl_loss = kl_loss.sum(dim=1).mean()

    # With beta=0, omit KL from the training objective but still report it.
    total_loss = reconstruction_loss if beta == 0 else reconstruction_loss + beta * kl_loss
    return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}


class ReacherVAE(nn.Module):
    

    def __init__(self, latent_dim: int = 8):
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 96 -> 48
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 48 -> 24
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 24 -> 12
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# 12 -> 6
            nn.ReLU(inplace=True),
            nn.Flatten(),                                        # (B, 256 * 6 * 6)
        )

        self.fc_mu = nn.Linear(256 * 6 * 6, latent_dim)
        self.fc_logvar = nn.Linear(256 * 6 * 6, latent_dim)

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 256 * 6 * 6),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 6, 6)),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 6 -> 12
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 12 -> 24
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 24 -> 48
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 48 -> 96
            nn.Sigmoid(),                                                  # pixels in [0, 1]
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Images -> latent mean and log variance, each (B, latent_dim)."""
        features = self.encoder(images)
        return self.fc_mu(features), self.fc_logvar(features)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + std * noise, preserving gradients through mu and std."""
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mu + std * noise

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent vectors (B, latent_dim) -> RGB images (B, 3, 96, 96)."""
        return self.decoder(self.decoder_input(z))

    def forward(
        self, images: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstruction, mean, and log variance for the VAE loss.

        Training uses sampled latents. Set sample=False to reconstruct from the
        mean for repeatable visual comparisons; eval() alone does not disable sampling.
        Generate new images by passing standard normal latent samples to decode().
        """
        mu, logvar = self.encode(images)
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(z), mu, logvar
