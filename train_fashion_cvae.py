"""Train and validate the Fashion-MNIST Conditional VAE."""

import argparse

import torch

from helper_lib.evaluator import evaluate_cvae_model
from helper_lib.fashion_cvae import get_fashion_cvae
from helper_lib.fashion_data import (
    FASHION_MNIST_CLASSES,
    get_fashion_mnist_loaders,
)
from helper_lib.trainer import train_cvae_model
from helper_lib.utils import get_device, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    """Parse Conditional VAE training options."""

    parser = argparse.ArgumentParser(
        description="Train the FashionGen Studio Conditional VAE."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--checkpoint-path",
        default="checkpoints/fashion_cvae.pth",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Train, validate, and save the Fashion-MNIST Conditional VAE."""

    args = parse_args()
    if args.latent_dim <= 0:
        raise ValueError("--latent-dim must be positive.")
    if args.beta < 0:
        raise ValueError("--beta cannot be negative.")

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_fashion_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        download=True,
    )
    model = get_fashion_cvae(
        latent_dim=args.latent_dim,
        num_classes=len(FASHION_MNIST_CLASSES),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = train_cvae_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        beta=args.beta,
        val_loader=val_loader,
    )
    validation_metrics = evaluate_cvae_model(
        model=model,
        data_loader=val_loader,
        device=device,
        beta=args.beta,
    )

    checkpoint_path = save_checkpoint(
        {
            "model_state_dict": model.state_dict(),
            "class_names": list(FASHION_MNIST_CLASSES),
            "latent_dim": args.latent_dim,
            "beta": args.beta,
            "training_history": history,
            "training_args": vars(args).copy(),
            "validation_metrics": validation_metrics,
        },
        args.checkpoint_path,
    )

    print(f"Final validation loss: {validation_metrics['loss']:.4f}")
    print(
        "Final validation components: "
        f"reconstruction={validation_metrics['reconstruction_loss']:.4f}, "
        f"kl={validation_metrics['kl_loss']:.4f}"
    )
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
