"""Evaluation helpers for FashionGen Studio classifiers and generators."""

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn

from helper_lib.fashion_cvae import cvae_loss_function


def evaluate_classifier_model(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device | str,
    criterion: nn.Module | None = None,
    class_names: Sequence[str] | None = None,
    compute_per_class: bool = True,
) -> dict[str, Any]:
    """Evaluate a classifier with sample-weighted loss and percentage accuracy."""

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct: torch.Tensor | None = None
    class_total: torch.Tensor | None = None

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            matches = predictions == labels
            correct += matches.sum().item()
            total += batch_size

            if compute_per_class:
                num_classes = logits.size(1)
                if class_correct is None or class_total is None:
                    class_correct = torch.zeros(num_classes, dtype=torch.long)
                    class_total = torch.zeros(num_classes, dtype=torch.long)
                cpu_labels = labels.detach().cpu()
                class_total += torch.bincount(cpu_labels, minlength=num_classes)
                class_correct += torch.bincount(
                    cpu_labels[matches.detach().cpu()],
                    minlength=num_classes,
                )

    if total == 0:
        raise ValueError("data_loader produced no samples.")

    per_class_accuracy: dict[str, dict[str, int | float | None]] = {}
    if compute_per_class and class_correct is not None and class_total is not None:
        num_classes = len(class_total)
        if class_names is not None and len(class_names) != num_classes:
            raise ValueError(
                f"Expected {num_classes} class names, received {len(class_names)}."
            )
        names = list(class_names) if class_names is not None else [
            str(index) for index in range(num_classes)
        ]
        for index, name in enumerate(names):
            class_count = int(class_total[index].item())
            class_hits = int(class_correct[index].item())
            class_accuracy = (
                100.0 * class_hits / class_count if class_count > 0 else None
            )
            per_class_accuracy[name] = {
                "correct": class_hits,
                "total": class_count,
                "accuracy": class_accuracy,
            }

    return {
        "loss": running_loss / total,
        "accuracy": 100.0 * correct / total,
        "per_class_accuracy": per_class_accuracy,
    }


def evaluate_cvae_model(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device | str,
    beta: float = 1.0,
) -> dict[str, float]:
    """Evaluate a Conditional VAE with sample-weighted average losses."""

    if beta < 0:
        raise ValueError("beta cannot be negative.")

    model.to(device)
    model.eval()
    running_total_loss = 0.0
    running_reconstruction_loss = 0.0
    running_kl_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            reconstruction, mean, logvar = model(images, labels)
            total_loss, reconstruction_loss, kl_loss = cvae_loss_function(
                reconstruction,
                images,
                mean,
                logvar,
                beta=beta,
            )

            batch_size = images.size(0)
            running_total_loss += total_loss.item() * batch_size
            running_reconstruction_loss += reconstruction_loss.item() * batch_size
            running_kl_loss += kl_loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("data_loader produced no samples.")

    return {
        "loss": running_total_loss / total_samples,
        "reconstruction_loss": running_reconstruction_loss / total_samples,
        "kl_loss": running_kl_loss / total_samples,
    }
