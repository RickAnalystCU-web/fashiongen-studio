"""Fashion-MNIST dataset constants and data-loading scaffold."""


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


def get_fashion_mnist_loaders(*args, **kwargs):
    """Build Fashion-MNIST train, validation, and test data loaders."""

    # TODO: Use torchvision.datasets.FashionMNIST with reproducible splits,
    # normalization, and configurable DataLoader settings.
    raise NotImplementedError("Fashion-MNIST data loading is not implemented yet.")
