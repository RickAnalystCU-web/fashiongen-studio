"""Conditional variational autoencoder for Fashion-MNIST generation."""

import torch
import torch.nn.functional as F
from torch import nn


class ConditionalVAE(nn.Module):
    """Convolutional VAE conditioned on Fashion-MNIST class labels."""

    def __init__(
        self,
        latent_dim: int = 32,
        num_classes: int = 10,
        label_embedding_dim: int = 16,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_embedding_dim = label_embedding_dim

        self.label_embedding = nn.Embedding(num_classes, label_embedding_dim)
        self.label_to_image = nn.Linear(label_embedding_dim, 28 * 28)

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim + label_embedding_dim, 64 * 7 * 7),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def _embed_labels(self, labels: torch.Tensor, batch_size: int) -> torch.Tensor:
        if labels.ndim != 1 or labels.size(0) != batch_size:
            raise ValueError(
                f"labels must have shape [{batch_size}], got {tuple(labels.shape)}."
            )
        return self.label_embedding(labels.long())

    def encode(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode images and labels into latent means and log variances."""

        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError(
                "images must have shape [batch, 1, 28, 28], "
                f"got {tuple(images.shape)}."
            )
        label_embeddings = self._embed_labels(labels, images.size(0))
        label_maps = self.label_to_image(label_embeddings).view(-1, 1, 28, 28)
        hidden = self.encoder(torch.cat((images, label_maps), dim=1))
        return self.mean_layer(hidden), self.logvar_layer(hidden)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample differentiable latent vectors with the reparameterization trick."""

        standard_deviation = torch.exp(0.5 * logvar)
        noise = torch.randn_like(standard_deviation)
        return mean + noise * standard_deviation

    def decode(self, latent: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors and labels into normalized fashion images."""

        if latent.ndim != 2 or latent.size(1) != self.latent_dim:
            raise ValueError(
                f"latent must have shape [batch, {self.latent_dim}], "
                f"got {tuple(latent.shape)}."
            )
        label_embeddings = self._embed_labels(labels, latent.size(0))
        hidden = self.decoder_input(torch.cat((latent, label_embeddings), dim=1))
        hidden = hidden.view(-1, 64, 7, 7)
        return self.decoder(hidden)

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstructions, latent means, and latent log variances."""

        mean, logvar = self.encode(images, labels)
        latent = self.reparameterize(mean, logvar)
        reconstruction = self.decode(latent, labels)
        return reconstruction, mean, logvar


def get_fashion_cvae(
    latent_dim: int = 32,
    num_classes: int = 10,
    label_embedding_dim: int = 16,
    hidden_dim: int = 256,
) -> ConditionalVAE:
    """Construct the standard FashionGen Studio Conditional VAE."""

    return ConditionalVAE(
        latent_dim=latent_dim,
        num_classes=num_classes,
        label_embedding_dim=label_embedding_dim,
        hidden_dim=hidden_dim,
    )


def cvae_loss_function(
    reconstruction: torch.Tensor,
    images: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, reconstruction, and KL losses averaged per sample."""

    if beta < 0:
        raise ValueError("beta cannot be negative.")
    if reconstruction.shape != images.shape:
        raise ValueError(
            "reconstruction and images must have the same shape, got "
            f"{tuple(reconstruction.shape)} and {tuple(images.shape)}."
        )

    batch_size = images.size(0)
    reconstruction_loss = F.mse_loss(
        reconstruction,
        images,
        reduction="sum",
    ) / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / batch_size
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss


def _shape_test() -> None:
    """Run a dataset-free CVAE shape and loss test."""

    latent_dim = 32
    model = get_fashion_cvae(latent_dim=latent_dim)
    images = torch.randn(4, 1, 28, 28).clamp(-1.0, 1.0)
    labels = torch.randint(0, 10, (4,))

    with torch.no_grad():
        reconstruction, mean, logvar = model(images, labels)
        total_loss, reconstruction_loss, kl_loss = cvae_loss_function(
            reconstruction,
            images,
            mean,
            logvar,
        )

    expected_image_shape = (4, 1, 28, 28)
    expected_latent_shape = (4, latent_dim)
    if tuple(reconstruction.shape) != expected_image_shape:
        raise RuntimeError(
            f"Expected reconstruction shape {expected_image_shape}, "
            f"got {tuple(reconstruction.shape)}."
        )
    if tuple(mean.shape) != expected_latent_shape:
        raise RuntimeError(
            f"Expected mean shape {expected_latent_shape}, got {tuple(mean.shape)}."
        )
    if tuple(logvar.shape) != expected_latent_shape:
        raise RuntimeError(
            f"Expected logvar shape {expected_latent_shape}, got {tuple(logvar.shape)}."
        )

    print(f"Reconstruction shape: {tuple(reconstruction.shape)}")
    print(f"Mean shape:           {tuple(mean.shape)}")
    print(f"Logvar shape:         {tuple(logvar.shape)}")
    print(
        "Losses: "
        f"total={total_loss.item():.4f}, "
        f"reconstruction={reconstruction_loss.item():.4f}, "
        f"kl={kl_loss.item():.4f}"
    )


if __name__ == "__main__":
    _shape_test()
