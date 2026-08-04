"""FastAPI entry point for FashionGen Studio."""

import io
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from helper_lib.fashion_data import FASHION_MNIST_CLASSES
from helper_lib.fashion_generator import (
    classify_fashion_tensors,
    generated_tensor_to_base64_png,
    get_cached_fashion_classifier,
    get_cached_fashion_cvae,
    preprocess_fashion_image,
    sample_fashion_images,
)


BASE_DIR = Path(__file__).resolve().parent
CVAE_CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "fashion_cvae.pth"
CLASSIFIER_CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "fashion_classifier.pth"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_GENERATED_IMAGES = 16

app = FastAPI(
    title="FashionGen Studio API",
    description=(
        "Generate class-conditioned Fashion-MNIST images with a Conditional VAE "
        "and evaluate images with a CNN quality checker."
    ),
    version="0.2.0",
)


class FashionGenerationRequest(BaseModel):
    """Request body shared by generation endpoints."""

    label: int | str = Field(
        ...,
        description="Fashion-MNIST class index or case-insensitive class name.",
        examples=["sneaker"],
    )
    num_images: int = Field(default=4, ge=1, le=MAX_GENERATED_IMAGES)
    seed: int | None = Field(default=42)


def _normalize_class_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _resolve_fashion_label(label: int | str) -> tuple[int, str]:
    if isinstance(label, bool):
        raise HTTPException(status_code=422, detail="Boolean labels are not valid.")

    if isinstance(label, int):
        class_index = label
    else:
        text = label.strip()
        if not text:
            raise HTTPException(status_code=422, detail="label cannot be empty.")
        if text.lstrip("-").isdigit():
            class_index = int(text)
        else:
            aliases = {
                _normalize_class_name(name): index
                for index, name in enumerate(FASHION_MNIST_CLASSES)
            }
            aliases.update({"tshirt": 0, "top": 0})
            normalized = _normalize_class_name(text)
            if normalized not in aliases:
                valid_names = ", ".join(FASHION_MNIST_CLASSES)
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown Fashion-MNIST label '{label}'. Valid names: {valid_names}.",
                )
            class_index = aliases[normalized]

    if not 0 <= class_index < len(FASHION_MNIST_CLASSES):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Fashion-MNIST class index must be between 0 and "
                f"{len(FASHION_MNIST_CLASSES) - 1}."
            ),
        )
    return class_index, FASHION_MNIST_CLASSES[class_index]


def _relative_checkpoint_path(path: Path) -> str:
    return path.relative_to(BASE_DIR).as_posix()


def _load_cvae_for_request():
    try:
        return get_cached_fashion_cvae(CVAE_CHECKPOINT_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"CVAE checkpoint not found at {_relative_checkpoint_path(CVAE_CHECKPOINT_PATH)}. "
                "Train it with: python train_fashion_cvae.py"
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"CVAE checkpoint could not be loaded: {exc}",
        ) from exc


def _load_classifier_for_request():
    try:
        return get_cached_fashion_classifier(CLASSIFIER_CHECKPOINT_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Classifier checkpoint not found at "
                f"{_relative_checkpoint_path(CLASSIFIER_CHECKPOINT_PATH)}. "
                "Train it with: python train_fashion_classifier.py"
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Classifier checkpoint could not be loaded: {exc}",
        ) from exc


@app.get("/")
def project_info() -> dict[str, Any]:
    """Return project metadata without loading either model checkpoint."""

    return {
        "project": "FashionGen Studio",
        "course": "APAN 5560",
        "status": "generation and CNN quality-check endpoints are available",
        "docs": "/docs",
        "endpoints": [
            "/fashion/classes",
            "/fashion/generate",
            "/fashion/analyze",
            "/fashion/generate-and-check",
        ],
    }


@app.get("/fashion/classes")
def fashion_classes() -> dict[str, Any]:
    """Return all canonical Fashion-MNIST class names and indices."""

    return {
        "dataset": "Fashion-MNIST",
        "count": len(FASHION_MNIST_CLASSES),
        "classes": [
            {"index": index, "name": name}
            for index, name in enumerate(FASHION_MNIST_CLASSES)
        ],
        "class_names": list(FASHION_MNIST_CLASSES),
    }


@app.post("/fashion/generate")
def generate_fashion(request: FashionGenerationRequest) -> dict[str, Any]:
    """Generate base64 PNG images for a requested Fashion-MNIST class."""

    class_index, class_name = _resolve_fashion_label(request.label)
    model, _, device = _load_cvae_for_request()
    images = sample_fashion_images(
        model=model,
        class_labels=[class_index] * request.num_images,
        device=device,
        seed=request.seed,
    )
    return {
        "model_name": "ConditionalVAE",
        "device": str(device),
        "requested_label": class_name,
        "class_index": class_index,
        "num_images": request.num_images,
        "seed": request.seed,
        "checkpoint_path": _relative_checkpoint_path(CVAE_CHECKPOINT_PATH),
        "image_format": "png",
        "images_base64": [
            generated_tensor_to_base64_png(image) for image in images
        ],
    }


@app.post("/fashion/analyze")
async def analyze_fashion_image(
    file: UploadFile = File(..., description="Fashion image to classify."),
) -> dict[str, Any]:
    """Classify an uploaded image with the trained Fashion-MNIST CNN."""

    try:
        image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Uploaded image exceeds the 5 MB limit.",
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as uploaded_image:
            image = uploaded_image.convert("L")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a readable image.",
        ) from exc

    model, checkpoint, device = _load_classifier_for_request()
    class_names = list(checkpoint.get("class_names", FASHION_MNIST_CLASSES))
    input_tensor = preprocess_fashion_image(image)
    prediction = classify_fashion_tensors(
        model=model,
        images=input_tensor,
        class_names=class_names,
        device=device,
        top_k=3,
    )[0]
    return {
        **prediction,
        "model_name": "FashionClassifier",
        "device": str(device),
        "checkpoint_path": _relative_checkpoint_path(CLASSIFIER_CHECKPOINT_PATH),
    }


@app.post("/fashion/generate-and-check")
def generate_and_check_fashion(
    request: FashionGenerationRequest,
) -> dict[str, Any]:
    """Generate class-conditioned images and validate each with the CNN."""

    class_index, class_name = _resolve_fashion_label(request.label)
    cvae, _, cvae_device = _load_cvae_for_request()
    classifier, classifier_checkpoint, classifier_device = (
        _load_classifier_for_request()
    )
    class_names = list(
        classifier_checkpoint.get("class_names", FASHION_MNIST_CLASSES)
    )

    generated_images = sample_fashion_images(
        model=cvae,
        class_labels=[class_index] * request.num_images,
        device=cvae_device,
        seed=request.seed,
    )
    predictions = classify_fashion_tensors(
        model=classifier,
        images=generated_images,
        class_names=class_names,
        device=classifier_device,
        top_k=3,
    )

    results = []
    for image, prediction in zip(generated_images, predictions, strict=True):
        results.append(
            {
                "image_base64": generated_tensor_to_base64_png(image),
                "requested_label": class_name,
                "predicted_label": prediction["predicted_label"],
                "predicted_class_index": prediction["predicted_class_index"],
                "confidence": prediction["confidence"],
                "passed_quality_check": (
                    prediction["predicted_class_index"] == class_index
                ),
            }
        )

    num_passed = sum(item["passed_quality_check"] for item in results)
    return {
        "requested_label": class_name,
        "requested_class_index": class_index,
        "num_images": request.num_images,
        "num_passed": num_passed,
        "pass_rate": num_passed / request.num_images,
        "seed": request.seed,
        "model_names": {
            "generator": "ConditionalVAE",
            "quality_checker": "FashionClassifier",
        },
        "device": str(cvae_device),
        "checkpoint_paths": {
            "generator": _relative_checkpoint_path(CVAE_CHECKPOINT_PATH),
            "quality_checker": _relative_checkpoint_path(
                CLASSIFIER_CHECKPOINT_PATH
            ),
        },
        "results": results,
    }
