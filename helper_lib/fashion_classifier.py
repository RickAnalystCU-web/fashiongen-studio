"""CNN classifier for 28 x 28 grayscale Fashion-MNIST images."""

import torch
from torch import nn


class FashionClassifier(nn.Module):
    """Compact CNN that returns raw logits for ten fashion classes."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits for a batch of images."""

        return self.classifier(self.features(images))


def get_fashion_classifier(
    num_classes: int = 10,
    dropout: float = 0.3,
) -> FashionClassifier:
    """Construct a Fashion-MNIST classifier with the standard architecture."""

    return FashionClassifier(num_classes=num_classes, dropout=dropout)


def _shape_test() -> None:
    """Run a small, dataset-free forward-pass shape test."""

    model = get_fashion_classifier()
    images = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        logits = model(images)

    expected_shape = (4, 10)
    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            f"Expected output shape {expected_shape}, got {tuple(logits.shape)}."
        )
    print(f"Input shape:  {tuple(images.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    _shape_test()
