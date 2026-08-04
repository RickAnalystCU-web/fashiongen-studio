"""Checkpoint loading and class-conditioned Fashion-MNIST generation helpers."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from helper_lib.fashion_cvae import ConditionalVAE, get_fashion_cvae
from helper_lib.fashion_data import FASHION_MNIST_CLASSES
from helper_lib.utils import get_device, image_to_base64_png, load_checkpoint


DEFAULT_CVAE_CHECKPOINT = Path("checkpoints") / "fashion_cvae.pth"


def load_fashion_cvae_checkpoint(
    checkpoint_path: str | Path = DEFAULT_CVAE_CHECKPOINT,
    device: torch.device | str | None = None,
) -> tuple[ConditionalVAE, dict[str, Any], torch.device]:
    """Load a trained CVAE and return the model, metadata, and device."""

    resolved_device = torch.device(device) if device is not None else get_device()
    checkpoint = load_checkpoint(checkpoint_path, device=resolved_device)
    if "model_state_dict" not in checkpoint:
        raise KeyError("CVAE checkpoint is missing model_state_dict.")

    class_names = checkpoint.get("class_names", FASHION_MNIST_CLASSES)
    latent_dim = int(checkpoint.get("latent_dim", 32))
    model = get_fashion_cvae(
        latent_dim=latent_dim,
        num_classes=len(class_names),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, checkpoint, resolved_device


def sample_fashion_images(
    model: ConditionalVAE,
    class_labels: Sequence[int] | torch.Tensor,
    device: torch.device | str | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample one normalized image for each requested Fashion-MNIST label."""

    resolved_device = torch.device(device) if device is not None else next(
        model.parameters()
    ).device
    labels = torch.as_tensor(class_labels, dtype=torch.long, device=resolved_device)
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("class_labels must be a non-empty one-dimensional sequence.")
    if labels.min().item() < 0 or labels.max().item() >= model.num_classes:
        raise ValueError(
            f"class labels must be between 0 and {model.num_classes - 1}."
        )

    random_generator = None
    if seed is not None:
        random_generator = torch.Generator(device=resolved_device).manual_seed(seed)
    latent = torch.randn(
        labels.size(0),
        model.latent_dim,
        device=resolved_device,
        generator=random_generator,
    )

    model.to(resolved_device)
    model.eval()
    with torch.no_grad():
        generated = model.decode(latent, labels)
    return generated.detach().cpu()


def generated_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert one normalized [-1, 1] tensor into an 8-bit grayscale image."""

    if image.ndim == 3 and image.size(0) == 1:
        image = image.squeeze(0)
    if image.ndim != 2 or tuple(image.shape) != (28, 28):
        raise ValueError(
            "image must have shape [1, 28, 28] or [28, 28], "
            f"got {tuple(image.shape)}."
        )

    display_tensor = ((image.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    image_array = (display_tensor * 255.0).round().to(torch.uint8).numpy()
    return Image.fromarray(image_array)


def generated_tensor_to_base64_png(image: torch.Tensor) -> str:
    """Convert one normalized generated tensor to a base64-encoded PNG."""

    return image_to_base64_png(generated_tensor_to_pil(image))


def generate_fashion_images(
    checkpoint_path: str | Path = DEFAULT_CVAE_CHECKPOINT,
    class_label: int = 0,
    num_images: int = 1,
    device: torch.device | str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Load a CVAE and generate base64 PNGs for one requested class."""

    if num_images <= 0:
        raise ValueError("num_images must be positive.")

    model, checkpoint, resolved_device = load_fashion_cvae_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    class_names = list(checkpoint.get("class_names", FASHION_MNIST_CLASSES))
    if not 0 <= class_label < len(class_names):
        raise ValueError(f"class_label must be between 0 and {len(class_names) - 1}.")

    images = sample_fashion_images(
        model=model,
        class_labels=[class_label] * num_images,
        device=resolved_device,
        seed=seed,
    )
    return {
        "class_id": class_label,
        "class_name": class_names[class_label],
        "image_format": "png",
        "images_base64": [
            generated_tensor_to_base64_png(image) for image in images
        ],
    }
