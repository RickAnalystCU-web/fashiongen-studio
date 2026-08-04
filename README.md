# FashionGen Studio

FashionGen Studio is the APAN 5560 final group project. It will generate one or more 28 x 28 grayscale fashion images for a requested category with a Conditional Variational Autoencoder (CVAE), then use a separate CNN classifier to check whether the generated image resembles the requested class. A FastAPI service will expose the workflow and return generated images as base64-encoded PNGs.

This repository currently contains the project scaffold only. Models have not been implemented or trained, and no dataset or checkpoint is included.

## Dataset

The project uses Fashion-MNIST through `torchvision.datasets.FashionMNIST`. Fashion-MNIST contains ten categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

Raw and downloaded data belongs under `data/`, which is intentionally ignored by Git. Dataset downloading will be added to the training workflow rather than performed during API startup.

## Planned models

- A CNN classifier trained on Fashion-MNIST as an independent class and confidence quality checker.
- A class-conditional VAE with label conditioning in the encoder and decoder.
- An evaluation workflow that compares the requested class with the CNN's prediction for each generated sample.

## Fashion-MNIST CNN quality checker

The CNN classifier provides an independent quality signal for generated images. It predicts one of the ten Fashion-MNIST classes from a 28 x 28 grayscale image and reports class accuracy metrics. In a later project step, it will evaluate CVAE samples returned by the planned `POST /fashion/generate-and-check` endpoint.

Train the classifier with the default five-epoch configuration:

```bash
uv run python train_fashion_classifier.py
```

For a short pipeline smoke test, use one epoch:

```bash
uv run python train_fashion_classifier.py --epochs 1
```

The default checkpoint is written to `checkpoints/fashion_classifier.pth`. It contains the model state, class names, test metrics, training arguments, and training history. Checkpoints and downloaded Fashion-MNIST data are ignored by Git and should not be committed.

## API endpoints

Available in the scaffold:

- `GET /` returns project metadata and implementation status.
- `GET /fashion/classes` returns the canonical Fashion-MNIST class names.

Planned model-backed endpoints:

- `POST /fashion/generate` will generate one or more images for a requested class and return base64 PNG data.
- `POST /fashion/quality-check` will classify an image and return the predicted class and confidence.
- `GET /health` will report API and checkpoint readiness.

Interactive API documentation is available at `/docs` while the app is running.

## Local development

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync
uv run uvicorn main:app --reload
```

Open `http://localhost:8000/docs`. A standard virtual environment also works:

```bash
python -m venv .venv
python -m pip install "fastapi[standard]" pillow torch torchvision tqdm "uvicorn[standard]"
uvicorn main:app --reload
```

## Docker

Build and run with Docker Compose:

```bash
docker compose up --build
```

Or use Docker directly:

```bash
docker build -t fashiongen-studio .
docker run --rm -p 8000:8000 fashiongen-studio
```

The API will be available at `http://localhost:8000`.

## Data, checkpoints, and generated output

- Do not commit `data/`; Fashion-MNIST should be downloaded locally by the future data-loading code.
- Do not commit trained weights (`*.pt`, `*.pth`, `*.ckpt`, or `*.onnx`). Keep local checkpoints under `checkpoints/` or use external artifact storage.
- `generated_samples/` is for local qualitative evaluation and is ignored except for its placeholder file.
- Never bake raw data or large model artifacts into the Docker image or Git history.

## Project layout

```text
fashiongen-studio/
|-- main.py
|-- helper_lib/
|   |-- fashion_data.py
|   |-- fashion_classifier.py
|   |-- fashion_cvae.py
|   |-- fashion_generator.py
|   |-- trainer.py
|   |-- evaluator.py
|   `-- utils.py
|-- train_fashion_classifier.py
|-- train_fashion_cvae.py
|-- checkpoints/
|-- generated_samples/
|-- report/
|-- slides/
|-- pyproject.toml
|-- Dockerfile
`-- docker-compose.yml
```
