"""Lightweight class-conditional diffusion utilities for Fashion-MNIST."""

import math
from collections.abc import Sequence

import torch
from torch import nn


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Encode scalar diffusion timesteps with sine and cosine frequencies."""

    if timesteps.ndim != 1:
        raise ValueError("timesteps must be one-dimensional.")
    half_dim = embedding_dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
        / max(half_dim - 1, 1)
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if embedding_dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


def cosine_beta_schedule(timesteps: int = 100, offset: float = 0.008) -> torch.Tensor:
    """Return a cosine schedule that reaches near-Gaussian terminal noise."""

    if timesteps <= 1:
        raise ValueError("timesteps must be greater than one.")
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    alpha_bars = torch.cos(
        ((steps / timesteps) + offset) / (1 + offset) * math.pi / 2
    ).pow(2)
    alpha_bars = alpha_bars / alpha_bars[0]
    return (1 - alpha_bars[1:] / alpha_bars[:-1]).clamp(0.0001, 0.999).float()


class DiffusionSchedule:
    """Precomputed DDPM coefficients for forward and reverse diffusion."""

    def __init__(
        self,
        timesteps: int = 100,
        device: torch.device | str | None = None,
    ) -> None:
        self.timesteps = timesteps
        self.device = torch.device(device or "cpu")
        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_previous = torch.cat(
            (torch.ones(1, device=self.device), self.alpha_bars[:-1])
        )
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alpha_bars_previous)
            / (1.0 - self.alpha_bars)
        ).clamp(min=1e-20)


def _extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    target_shape: torch.Size | tuple[int, ...],
) -> torch.Tensor:
    """Gather schedule values and reshape them for image broadcasting."""

    gathered = values.gather(0, timesteps.long())
    return gathered.reshape(timesteps.size(0), *((1,) * (len(target_shape) - 1)))


def q_sample(
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the closed-form forward process q(x_t | x_0)."""

    if clean_images.ndim != 4 or tuple(clean_images.shape[1:]) != (1, 28, 28):
        raise ValueError(
            "clean_images must have shape [batch, 1, 28, 28], "
            f"got {tuple(clean_images.shape)}."
        )
    if timesteps.ndim != 1 or timesteps.size(0) != clean_images.size(0):
        raise ValueError("timesteps must match the image batch.")
    if timesteps.min().item() < 0 or timesteps.max().item() >= schedule.timesteps:
        raise ValueError("timesteps contain a value outside the schedule.")
    sampled_noise = torch.randn_like(clean_images) if noise is None else noise
    if sampled_noise.shape != clean_images.shape:
        raise ValueError("noise and clean_images must have the same shape.")
    clean_scale = _extract(
        schedule.sqrt_alpha_bars, timesteps, clean_images.shape
    )
    noise_scale = _extract(
        schedule.sqrt_one_minus_alpha_bars, timesteps, clean_images.shape
    )
    return clean_scale * clean_images + noise_scale * sampled_noise, sampled_noise


class ConditionalResidualBlock(nn.Module):
    """Residual convolutional block modulated by time and class context."""

    def __init__(self, in_channels: int, out_channels: int, context_dim: int) -> None:
        super().__init__()
        self.norm_1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.context_projection = nn.Linear(context_dim, out_channels)
        self.norm_2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.activation = nn.SiLU()
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, images: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = self.conv_1(self.activation(self.norm_1(images)))
        hidden = hidden + self.context_projection(context)[:, :, None, None]
        hidden = self.conv_2(self.activation(self.norm_2(hidden)))
        return hidden + self.residual(images)


class SmallConditionalUNet(nn.Module):
    """Small UNet-like epsilon predictor for 28 x 28 grayscale images."""

    def __init__(self, base_channels: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        if base_channels < 8 or base_channels % 8 != 0:
            raise ValueError("base_channels must be at least 8 and divisible by 8.")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")

        self.base_channels = base_channels
        self.num_classes = num_classes
        context_dim = base_channels * 4
        self.timestep_mlp = nn.Sequential(
            nn.Linear(base_channels, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )
        self.label_embedding = nn.Embedding(num_classes, context_dim)
        self.input_conv = nn.Conv2d(1, base_channels, 3, padding=1)
        self.down_block_1 = ConditionalResidualBlock(
            base_channels, base_channels, context_dim
        )
        self.downsample_1 = nn.Conv2d(
            base_channels, base_channels * 2, 4, stride=2, padding=1
        )
        self.down_block_2 = ConditionalResidualBlock(
            base_channels * 2, base_channels * 2, context_dim
        )
        self.downsample_2 = nn.Conv2d(
            base_channels * 2, base_channels * 4, 4, stride=2, padding=1
        )
        self.middle_block_1 = ConditionalResidualBlock(
            base_channels * 4, base_channels * 4, context_dim
        )
        self.middle_block_2 = ConditionalResidualBlock(
            base_channels * 4, base_channels * 4, context_dim
        )
        self.upsample_1 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, 4, stride=2, padding=1
        )
        self.up_block_1 = ConditionalResidualBlock(
            base_channels * 4, base_channels * 2, context_dim
        )
        self.upsample_2 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, 4, stride=2, padding=1
        )
        self.up_block_2 = ConditionalResidualBlock(
            base_channels * 2, base_channels, context_dim
        )
        self.output = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 1, 3, padding=1),
        )

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the Gaussian noise added at each supplied timestep."""

        batch_size = noisy_images.size(0)
        if noisy_images.ndim != 4 or tuple(noisy_images.shape[1:]) != (1, 28, 28):
            raise ValueError(
                "noisy_images must have shape [batch, 1, 28, 28], "
                f"got {tuple(noisy_images.shape)}."
            )
        if timesteps.ndim != 1 or timesteps.size(0) != batch_size:
            raise ValueError(f"timesteps must have shape [{batch_size}].")
        if labels.ndim != 1 or labels.size(0) != batch_size:
            raise ValueError(f"labels must have shape [{batch_size}].")
        time_features = sinusoidal_timestep_embedding(
            timesteps, self.base_channels
        )
        context = self.timestep_mlp(time_features) + self.label_embedding(labels.long())
        input_features = self.input_conv(noisy_images)
        skip_1 = self.down_block_1(input_features, context)
        skip_2 = self.down_block_2(self.downsample_1(skip_1), context)
        hidden = self.middle_block_1(self.downsample_2(skip_2), context)
        hidden = self.middle_block_2(hidden, context)
        hidden = self.up_block_1(
            torch.cat((self.upsample_1(hidden), skip_2), dim=1), context
        )
        hidden = self.up_block_2(
            torch.cat((self.upsample_2(hidden), skip_1), dim=1), context
        )
        return self.output(hidden)


FashionDiffusionDenoiser = SmallConditionalUNet


def get_fashion_diffusion_model(
    base_channels: int = 32,
    num_classes: int = 10,
) -> SmallConditionalUNet:
    """Construct the standard lightweight FashionGen diffusion denoiser."""

    return SmallConditionalUNet(base_channels, num_classes)


@torch.no_grad()
def p_sample(
    model: SmallConditionalUNet,
    noisy_images: torch.Tensor,
    timesteps: torch.Tensor,
    labels: torch.Tensor,
    schedule: DiffusionSchedule,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Perform one DDPM reverse step p(x_(t-1) | x_t, y)."""

    predicted_noise = model(noisy_images, timesteps, labels)
    beta_t = _extract(schedule.betas, timesteps, noisy_images.shape)
    alpha_t = _extract(schedule.alphas, timesteps, noisy_images.shape)
    alpha_bar_t = _extract(schedule.alpha_bars, timesteps, noisy_images.shape)
    model_mean = (
        noisy_images
        - beta_t / torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
    ) / torch.sqrt(alpha_t)
    random_noise = torch.randn(
        noisy_images.shape,
        device=noisy_images.device,
        dtype=noisy_images.dtype,
        generator=generator,
    )
    posterior_variance = _extract(
        schedule.posterior_variance, timesteps, noisy_images.shape
    )
    nonzero = (timesteps != 0).float().reshape(
        timesteps.size(0), *((1,) * (noisy_images.ndim - 1))
    )
    return model_mean + nonzero * torch.sqrt(posterior_variance) * random_noise


@torch.no_grad()
def generate_diffusion_samples(
    model: SmallConditionalUNet,
    class_labels: Sequence[int] | torch.Tensor,
    schedule: DiffusionSchedule,
    device: torch.device | str | None = None,
    seed: int = 42,
) -> torch.Tensor:
    """Generate one normalized image per requested class with reverse diffusion."""

    resolved_device = torch.device(device) if device is not None else next(
        model.parameters()
    ).device
    labels = torch.as_tensor(class_labels, dtype=torch.long, device=resolved_device)
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("class_labels must be a non-empty one-dimensional sequence.")
    if labels.min().item() < 0 or labels.max().item() >= model.num_classes:
        raise ValueError(f"class labels must be between 0 and {model.num_classes - 1}.")
    if schedule.device != resolved_device:
        raise ValueError("schedule and sampling devices must match.")
    generator = torch.Generator(device=resolved_device).manual_seed(seed)
    images = torch.randn(
        labels.size(0), 1, 28, 28, device=resolved_device, generator=generator
    )
    model.to(resolved_device)
    model.eval()
    for timestep in reversed(range(schedule.timesteps)):
        batch_timesteps = torch.full(
            (labels.size(0),), timestep, device=resolved_device, dtype=torch.long
        )
        images = p_sample(
            model,
            images,
            batch_timesteps,
            labels,
            schedule,
            generator,
        )
    return images.clamp(-1.0, 1.0).cpu()


def _shape_test() -> None:
    """Run a dataset-free diffusion and denoiser shape test."""

    model = get_fashion_diffusion_model()
    schedule = DiffusionSchedule(100)
    images = torch.randn(4, 1, 28, 28).clamp(-1.0, 1.0)
    timesteps = torch.randint(0, 100, (4,))
    labels = torch.randint(0, 10, (4,))
    noisy_images, noise = q_sample(images, timesteps, schedule)
    predicted_noise = model(noisy_images, timesteps, labels)
    reverse_images = p_sample(model, noisy_images, timesteps, labels, schedule)
    expected_shape = (4, 1, 28, 28)
    for name, tensor in (
        ("noisy images", noisy_images),
        ("noise", noise),
        ("predicted noise", predicted_noise),
        ("reverse images", reverse_images),
    ):
        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(f"Unexpected {name} shape: {tuple(tensor.shape)}")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Noisy image shape:     {tuple(noisy_images.shape)}")
    print(f"Predicted noise shape: {tuple(predicted_noise.shape)}")
    print(f"Reverse image shape:   {tuple(reverse_images.shape)}")
    print(f"Trainable parameters:  {parameters:,}")


if __name__ == "__main__":
    _shape_test()
