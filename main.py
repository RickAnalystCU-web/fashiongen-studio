"""FastAPI entry point for FashionGen Studio."""

from fastapi import FastAPI

from helper_lib.fashion_data import FASHION_MNIST_CLASSES


app = FastAPI(
    title="FashionGen Studio API",
    description=(
        "Conditional Fashion-MNIST generation with CNN-based quality checking. "
        "Model-backed endpoints will be added after training is implemented."
    ),
    version="0.1.0",
)


@app.get("/")
def project_info() -> dict[str, str]:
    """Return static project metadata for the initial scaffold."""

    return {
        "project": "FashionGen Studio",
        "course": "APAN 5560",
        "status": "scaffold initialized; models are not trained",
        "docs": "/docs",
    }


@app.get("/fashion/classes")
def fashion_classes() -> dict[str, object]:
    """Return the canonical Fashion-MNIST class labels."""

    return {
        "dataset": "Fashion-MNIST",
        "count": len(FASHION_MNIST_CLASSES),
        "classes": FASHION_MNIST_CLASSES,
    }
