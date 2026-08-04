"""Train and evaluate the Fashion-MNIST CNN quality checker."""

import argparse

import torch
from torch import nn

from helper_lib.evaluator import evaluate_classifier_model
from helper_lib.fashion_classifier import get_fashion_classifier
from helper_lib.fashion_data import (
    FASHION_MNIST_CLASSES,
    get_fashion_mnist_loaders,
)
from helper_lib.trainer import train_classifier_model
from helper_lib.utils import get_device, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    """Parse classifier training options."""

    parser = argparse.ArgumentParser(
        description="Train the FashionGen Studio Fashion-MNIST classifier."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--checkpoint-path",
        default="checkpoints/fashion_classifier.pth",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Train, validate, test, and save the Fashion-MNIST classifier."""

    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_fashion_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        download=True,
    )

    model = get_fashion_classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = train_classifier_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        criterion=criterion,
        val_loader=val_loader,
    )
    test_metrics = evaluate_classifier_model(
        model=model,
        data_loader=test_loader,
        device=device,
        criterion=criterion,
        class_names=FASHION_MNIST_CLASSES,
        compute_per_class=True,
    )

    checkpoint_path = save_checkpoint(
        {
            "model_state_dict": model.state_dict(),
            "class_names": list(FASHION_MNIST_CLASSES),
            "test_metrics": test_metrics,
            "training_args": vars(args).copy(),
            "training_history": history,
        },
        args.checkpoint_path,
    )

    print(f"Final test loss: {test_metrics['loss']:.4f}")
    print(f"Final test accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
