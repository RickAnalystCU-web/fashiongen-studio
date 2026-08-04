"""Fashion-MNIST constants, transforms, and reproducible data loaders."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def get_fashion_mnist_transform() -> transforms.Compose:
    """Return the shared transform that maps pixels from [0, 1] to [-1, 1]."""

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def get_fashion_mnist_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    validation_split: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create reproducible Fashion-MNIST train, validation, and test loaders."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    transform = get_fashion_mnist_transform()
    data_path = Path(data_dir)

    full_train_dataset = datasets.FashionMNIST(
        root=data_path,
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=data_path,
        train=False,
        download=download,
        transform=transform,
    )

    validation_size = int(len(full_train_dataset) * validation_split)
    train_size = len(full_train_dataset) - validation_size
    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [train_size, validation_size],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, validation_loader, test_loader


def _main() -> None:
    """Print dataset and one-batch details as a small data-pipeline test."""

    parser = argparse.ArgumentParser(description="Test the Fashion-MNIST loaders.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require Fashion-MNIST to already exist in data-dir.",
    )
    args = parser.parse_args()

    train_loader, validation_loader, test_loader = get_fashion_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=not args.no_download,
    )
    images, labels = next(iter(train_loader))

    print(f"Train size:      {len(train_loader.dataset)}")
    print(f"Validation size: {len(validation_loader.dataset)}")
    print(f"Test size:       {len(test_loader.dataset)}")
    print(f"Batch images:    {tuple(images.shape)}")
    print(f"Batch labels:    {tuple(labels.shape)}")
    print(f"Class names:     {FASHION_MNIST_CLASSES}")


if __name__ == "__main__":
    _main()
