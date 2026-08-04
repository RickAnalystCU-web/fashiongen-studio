"""Reusable training loops for FashionGen Studio models."""

from collections.abc import Iterable

import torch
from torch import nn

from helper_lib.evaluator import evaluate_classifier_model, evaluate_cvae_model
from helper_lib.fashion_cvae import cvae_loss_function


def train_classifier_model(
    model: nn.Module,
    train_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    epochs: int = 5,
    criterion: nn.Module | None = None,
    val_loader: Iterable | None = None,
) -> dict[str, list[float]]:
    """Train a classifier and return sample-weighted loss and accuracy history.

    Accuracy values are percentages in the range [0, 100]. Validation history
    lists remain empty when ``val_loader`` is not provided.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.to(device)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

        if total == 0:
            raise ValueError("train_loader produced no samples.")

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        message = (
            f"Epoch [{epoch + 1}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_accuracy={train_accuracy:.2f}%"
        )

        if val_loader is not None:
            validation_metrics = evaluate_classifier_model(
                model=model,
                data_loader=val_loader,
                device=device,
                criterion=criterion,
                compute_per_class=False,
            )
            history["val_loss"].append(validation_metrics["loss"])
            history["val_accuracy"].append(validation_metrics["accuracy"])
            message += (
                f" val_loss={validation_metrics['loss']:.4f}"
                f" val_accuracy={validation_metrics['accuracy']:.2f}%"
            )

        print(message)

    return history


def train_cvae_model(
    model: nn.Module,
    train_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    epochs: int = 10,
    beta: float = 1.0,
    val_loader: Iterable | None = None,
) -> dict[str, list[float]]:
    """Train a Conditional VAE and return sample-weighted loss history."""

    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if beta < 0:
        raise ValueError("beta cannot be negative.")

    model.to(device)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_reconstruction_loss": [],
        "train_kl_loss": [],
        "val_loss": [],
        "val_reconstruction_loss": [],
        "val_kl_loss": [],
    }

    for epoch in range(epochs):
        model.train()
        running_total_loss = 0.0
        running_reconstruction_loss = 0.0
        running_kl_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            reconstruction, mean, logvar = model(images, labels)
            total_loss, reconstruction_loss, kl_loss = cvae_loss_function(
                reconstruction,
                images,
                mean,
                logvar,
                beta=beta,
            )
            total_loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_total_loss += total_loss.item() * batch_size
            running_reconstruction_loss += reconstruction_loss.item() * batch_size
            running_kl_loss += kl_loss.item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("train_loader produced no samples.")

        train_loss = running_total_loss / total_samples
        train_reconstruction_loss = running_reconstruction_loss / total_samples
        train_kl_loss = running_kl_loss / total_samples
        history["train_loss"].append(train_loss)
        history["train_reconstruction_loss"].append(train_reconstruction_loss)
        history["train_kl_loss"].append(train_kl_loss)

        message = (
            f"Epoch [{epoch + 1}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_reconstruction={train_reconstruction_loss:.4f} "
            f"train_kl={train_kl_loss:.4f}"
        )

        if val_loader is not None:
            validation_metrics = evaluate_cvae_model(
                model=model,
                data_loader=val_loader,
                device=device,
                beta=beta,
            )
            history["val_loss"].append(validation_metrics["loss"])
            history["val_reconstruction_loss"].append(
                validation_metrics["reconstruction_loss"]
            )
            history["val_kl_loss"].append(validation_metrics["kl_loss"])
            message += (
                f" val_loss={validation_metrics['loss']:.4f}"
                f" val_reconstruction="
                f"{validation_metrics['reconstruction_loss']:.4f}"
                f" val_kl={validation_metrics['kl_loss']:.4f}"
            )

        print(message)

    return history
