"""Small shared utilities for FashionGen Studio."""

import base64
import io
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def image_to_base64_png(image: Image.Image) -> str:
    """Encode a Pillow image as a UTF-8 base64 PNG string."""

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python and PyTorch random number generators."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    checkpoint: Mapping[str, Any],
    checkpoint_path: str | Path,
) -> Path:
    """Create the parent directory and save a PyTorch checkpoint."""

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(checkpoint), path)
    return path


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Load a dictionary checkpoint using safe weights-only deserialization."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(
        path,
        map_location=device or get_device(),
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a dictionary checkpoint, got {type(checkpoint)!r}.")
    return checkpoint


# TODO: Add tensor-to-Pillow conversion after the CVAE output range and
# normalization convention are finalized.
