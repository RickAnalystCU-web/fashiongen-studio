"""Lazy model loading, inference, and Fashion-MNIST image generation helpers."""

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from helper_lib.fashion_classifier import (
    FashionClassifier,
    get_fashion_classifier,
)
from helper_lib.fashion_cvae import ConditionalVAE, get_fashion_cvae
from helper_lib.fashion_data import FASHION_MNIST_CLASSES
from helper_lib.utils import get_device, image_to_base64_png, load_checkpoint


DEFAULT_CVAE_CHECKPOINT = Path("checkpoints") / "fashion_cvae.pth"
DEFAULT_CLASSIFIER_CHECKPOINT = Path("checkpoints") / "fashion_classifier.pth"

FASHION_CLASSIFIER_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def load_fashion_cvae_checkpoint(
    checkpoint_path: str | Path = DEFAULT_CVAE_CHECKPOINT,
    device: torch.device | str | None = None,
) -> tuple[ConditionalVAE, dict[str, Any], torch.device]:
    """Load a trained CVAE and return the model, metadata, and device."""

    resolved_device = torch.device(device) if device is not None else get_device()
    checkpoint = load_checkpoint(checkpoint_path, device=resolved_device)
    if "model_state_dict" not in checkpoint:
        raise KeyError("CVAE checkpoint is missing model_state_dict.")

    class_names = list(checkpoint.get("class_names", FASHION_MNIST_CLASSES))
    latent_dim = int(checkpoint.get("latent_dim", 32))
    model = get_fashion_cvae(
        latent_dim=latent_dim,
        num_classes=len(class_names),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, checkpoint, resolved_device


@lru_cache(maxsize=4)
def _cached_cvae_loader(
    checkpoint_path: str,
    device: str,
) -> tuple[ConditionalVAE, dict[str, Any], torch.device]:
    return load_fashion_cvae_checkpoint(checkpoint_path, device=device)


def get_cached_fashion_cvae(
    checkpoint_path: str | Path = DEFAULT_CVAE_CHECKPOINT,
    device: torch.device | str | None = None,
) -> tuple[ConditionalVAE, dict[str, Any], torch.device]:
    """Load the CVAE once per resolved checkpoint path and device."""

    resolved_path = str(Path(checkpoint_path).resolve())
    resolved_device = str(torch.device(device) if device is not None else get_device())
    return _cached_cvae_loader(resolved_path, resolved_device)


def load_fashion_classifier_checkpoint(
    checkpoint_path: str | Path = DEFAULT_CLASSIFIER_CHECKPOINT,
    device: torch.device | str | None = None,
) -> tuple[FashionClassifier, dict[str, Any], torch.device]:
    """Load the trained Fashion-MNIST classifier and its metadata."""

    resolved_device = torch.device(device) if device is not None else get_device()
    checkpoint = load_checkpoint(checkpoint_path, device=resolved_device)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Classifier checkpoint is missing model_state_dict.")

    class_names = list(checkpoint.get("class_names", FASHION_MNIST_CLASSES))
    model = get_fashion_classifier(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, checkpoint, resolved_device


@lru_cache(maxsize=4)
def _cached_classifier_loader(
    checkpoint_path: str,
    device: str,
) -> tuple[FashionClassifier, dict[str, Any], torch.device]:
    return load_fashion_classifier_checkpoint(checkpoint_path, device=device)


def get_cached_fashion_classifier(
    checkpoint_path: str | Path = DEFAULT_CLASSIFIER_CHECKPOINT,
    device: torch.device | str | None = None,
) -> tuple[FashionClassifier, dict[str, Any], torch.device]:
    """Load the classifier once per resolved checkpoint path and device."""

    resolved_path = str(Path(checkpoint_path).resolve())
    resolved_device = str(torch.device(device) if device is not None else get_device())
    return _cached_classifier_loader(resolved_path, resolved_device)


def clear_model_caches() -> None:
    """Clear lazy model caches, primarily for tests or replaced checkpoints."""

    _cached_cvae_loader.cache_clear()
    _cached_classifier_loader.cache_clear()


def preprocess_fashion_image(image: Image.Image) -> torch.Tensor:
    """Convert an uploaded image into a normalized [1, 28, 28] tensor."""

    return FASHION_CLASSIFIER_TRANSFORM(image.convert("L"))


def classify_fashion_tensors(
    model: FashionClassifier,
    images: torch.Tensor,
    class_names: Sequence[str] = FASHION_MNIST_CLASSES,
    device: torch.device | str | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Classify normalized Fashion-MNIST tensors and return ranked predictions."""

    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
        raise ValueError(
            "images must have shape [batch, 1, 28, 28], "
            f"got {tuple(images.shape)}."
        )
    if len(class_names) == 0:
        raise ValueError("class_names cannot be empty.")

    resolved_device = torch.device(device) if device is not None else next(
        model.parameters()
    ).device
    model.to(resolved_device)
    model.eval()
    with torch.no_grad():
        logits = model(images.to(resolved_device))
        probabilities = torch.softmax(logits, dim=1)

    ranked_count = min(max(top_k, 1), probabilities.size(1))
    top_probabilities, top_indices = probabilities.topk(ranked_count, dim=1)
    results: list[dict[str, Any]] = []
    for row in range(probabilities.size(0)):
        predicted_index = int(top_indices[row, 0].item())
        top_predictions = [
            {
                "label": class_names[int(index.item())],
                "class_index": int(index.item()),
                "confidence": float(probability.item()),
            }
            for probability, index in zip(
                top_probabilities[row],
                top_indices[row],
                strict=True,
            )
        ]
        results.append(
            {
                "predicted_label": class_names[predicted_index],
                "predicted_class_index": predicted_index,
                "confidence": float(top_probabilities[row, 0].item()),
                "top_3": top_predictions,
            }
        )
    return results


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
    """Load a cached CVAE and generate base64 PNGs for one requested class."""

    if num_images <= 0:
        raise ValueError("num_images must be positive.")

    model, checkpoint, resolved_device = get_cached_fashion_cvae(
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

def interpolate_latent_vectors(
    start_latent: torch.Tensor,
    end_latent: torch.Tensor,
    steps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly interpolate between two one-dimensional latent vectors."""

    if steps < 2:
        raise ValueError("steps must be at least 2.")
    if start_latent.ndim != 1 or end_latent.ndim != 1:
        raise ValueError("start_latent and end_latent must be one-dimensional.")
    if start_latent.shape != end_latent.shape:
        raise ValueError("start_latent and end_latent must have matching shapes.")

    alphas = torch.linspace(
        0.0,
        1.0,
        steps,
        device=start_latent.device,
        dtype=start_latent.dtype,
    )
    interpolation = (
        (1.0 - alphas).unsqueeze(1) * start_latent.unsqueeze(0)
        + alphas.unsqueeze(1) * end_latent.unsqueeze(0)
    )
    return interpolation, alphas


def sample_fashion_latent_interpolation(
    model: ConditionalVAE,
    start_label: int,
    end_label: int,
    steps: int = 8,
    device: torch.device | str | None = None,
    seed: int | None = 42,
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Decode a seeded linear latent path with a midpoint label switch.

    The start class conditions the first half of the path and the end class
    conditions the second half. Returned images and interpolation coefficients
    are moved to CPU for API serialization and artifact generation.
    """

    if steps < 2:
        raise ValueError("steps must be at least 2.")
    if not 0 <= start_label < model.num_classes:
        raise ValueError(
            f"start_label must be between 0 and {model.num_classes - 1}."
        )
    if not 0 <= end_label < model.num_classes:
        raise ValueError(
            f"end_label must be between 0 and {model.num_classes - 1}."
        )

    resolved_device = torch.device(device) if device is not None else next(
        model.parameters()
    ).device
    random_generator = None
    if seed is not None:
        random_generator = torch.Generator(device=resolved_device).manual_seed(seed)
    latent_anchors = torch.randn(
        2,
        model.latent_dim,
        device=resolved_device,
        generator=random_generator,
    )
    latent_path, alphas = interpolate_latent_vectors(
        latent_anchors[0],
        latent_anchors[1],
        steps=steps,
    )

    split_point = (steps + 1) // 2
    label_schedule = [start_label] * split_point + [end_label] * (
        steps - split_point
    )
    labels = torch.tensor(
        label_schedule,
        dtype=torch.long,
        device=resolved_device,
    )

    model.to(resolved_device)
    model.eval()
    with torch.no_grad():
        generated = model.decode(latent_path, labels)
    return generated.detach().cpu(), label_schedule, alphas.detach().cpu()


def interpolation_tensors_to_base64(images: torch.Tensor) -> list[str]:
    """Encode a batch of normalized interpolation images as base64 PNGs."""

    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
        raise ValueError(
            "images must have shape [steps, 1, 28, 28], "
            f"got {tuple(images.shape)}."
        )
    return [generated_tensor_to_base64_png(image) for image in images]


def generate_fashion_interpolation(
    checkpoint_path: str | Path = DEFAULT_CVAE_CHECKPOINT,
    start_label: int = 7,
    end_label: int = 9,
    steps: int = 8,
    device: torch.device | str | None = None,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Load the cached CVAE and return one base64-encoded latent path."""

    model, checkpoint, resolved_device = get_cached_fashion_cvae(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    class_names = list(checkpoint.get("class_names", FASHION_MNIST_CLASSES))
    if not 0 <= start_label < len(class_names):
        raise ValueError(f"start_label must be between 0 and {len(class_names) - 1}.")
    if not 0 <= end_label < len(class_names):
        raise ValueError(f"end_label must be between 0 and {len(class_names) - 1}.")

    images, label_schedule, alphas = sample_fashion_latent_interpolation(
        model=model,
        start_label=start_label,
        end_label=end_label,
        steps=steps,
        device=resolved_device,
        seed=seed,
    )
    return {
        "start_class_index": start_label,
        "start_label": class_names[start_label],
        "end_class_index": end_label,
        "end_label": class_names[end_label],
        "steps": steps,
        "seed": seed,
        "label_schedule": [class_names[index] for index in label_schedule],
        "alphas": [float(alpha) for alpha in alphas],
        "image_format": "png",
        "images_base64": interpolation_tensors_to_base64(images),
        "model_name": "ConditionalVAE",
        "device": str(resolved_device),
    }
