# FashionGen Studio

FashionGen Studio is the APAN 5560 final group project. It generates one or more 28 x 28 grayscale fashion images for a requested category with a Conditional Variational Autoencoder (CVAE), then uses a separate CNN classifier to check whether each generated image resembles the requested class. A FastAPI service exposes the workflow and returns generated images as base64-encoded PNGs.

The repository includes trained-model pipelines and user-facing FastAPI endpoints. Raw data and trained checkpoints remain local and are intentionally excluded from Git.

## Dataset

The project uses Fashion-MNIST through `torchvision.datasets.FashionMNIST`. Fashion-MNIST contains ten categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

Raw and downloaded data belongs under `data/`, which is intentionally ignored by Git. The training scripts download Fashion-MNIST on demand; API startup never downloads the dataset.

## Models

- A CNN classifier trained on Fashion-MNIST as an independent class and confidence quality checker.
- A class-conditional VAE with label conditioning in the encoder and decoder.
- An evaluation workflow that compares the requested class with the CNN's prediction for each generated sample.

## Fashion-MNIST CNN quality checker

The CNN classifier provides an independent quality signal for generated images. It predicts one of the ten Fashion-MNIST classes from a 28 x 28 grayscale image and reports class accuracy metrics. It also evaluates CVAE samples returned by the implemented `POST /fashion/generate-and-check` endpoint.

Train the classifier with the default five-epoch configuration:

```bash
uv run python train_fashion_classifier.py
```

For a short pipeline smoke test, use one epoch:

```bash
uv run python train_fashion_classifier.py --epochs 1
```

The default checkpoint is written to `checkpoints/fashion_classifier.pth`. It contains the model state, class names, test metrics, training arguments, and training history. Checkpoints and downloaded Fashion-MNIST data are ignored by Git and should not be committed.

## Conditional VAE generator

The Conditional VAE (CVAE) learns a class-conditioned latent representation of normalized Fashion-MNIST images. A requested class label is embedded and supplied to both the convolutional encoder and transposed-convolution decoder. The decoder uses a `Tanh` output so generated pixels match the training range of approximately [-1, 1]. Training minimizes per-image reconstruction loss plus beta-weighted KL divergence.

Train the default model with a 32-dimensional latent space and beta of 1.0:

```bash
uv run python train_fashion_cvae.py
```

Training options include `--epochs`, `--batch-size`, `--lr`, `--latent-dim`, and `--beta`. The default checkpoint is written to `checkpoints/fashion_cvae.pth` and remains ignored by Git.

Standalone helpers in `helper_lib/fashion_generator.py` can load the checkpoint, sample requested classes, and encode generated grayscale images as base64 PNGs. These helpers support the FastAPI generation and generation-with-quality-check endpoints.

## API endpoints

Interactive Swagger documentation is available at `http://localhost:8000/docs`. Model checkpoints are loaded lazily, so the API can start without them; a model-backed endpoint returns a helpful error if its required checkpoint is missing.

Required local checkpoints:

- `checkpoints/fashion_classifier.pth`
- `checkpoints/fashion_cvae.pth`

Recreate them with:

```bash
uv run python train_fashion_classifier.py
uv run python train_fashion_cvae.py
```

### `GET /`

Returns project metadata and the available user-facing routes.

### `GET /fashion/classes`

Returns all ten Fashion-MNIST classes with their numeric indices and canonical names.

### `POST /fashion/generate`

Generates 1 to 16 class-conditioned images and returns base64-encoded PNG strings. `label` accepts a class name or index.

```json
{
  "label": "sneaker",
  "num_images": 4,
  "seed": 42
}
```

### `POST /fashion/analyze`

Accepts an uploaded image as multipart form data. The service converts it to grayscale, resizes it to 28 x 28, applies Fashion-MNIST normalization, and returns the CNN prediction, confidence, and top-three classes.

```bash
curl -X POST "http://localhost:8000/fashion/analyze" \
  -F "file=@generated_sample.png"
```

### `POST /fashion/generate-and-check`

Generates images with the CVAE and immediately classifies each normalized tensor with the CNN. Each result includes the PNG, requested and predicted labels, confidence, and a pass/fail quality flag. The summary includes the number passed and pass rate.

```json
{
  "label": "bag",
  "num_images": 4,
  "seed": 42
}
```

## Report and presentation outputs

After both trained checkpoints are available locally, generate the final evaluation artifacts with:

```bash
uv run python generate_project_outputs.py
```

The default run uses seed 42 and generates eight samples for each of the ten Fashion-MNIST classes. It creates:

- `generated_samples/cvae_all_classes_grid.png` — a labeled ten-row image grid with green classifier-agreement borders and red mismatch borders.
- `generated_samples/generate_and_check_summary.csv` — one requested-versus-predicted classifier result per generated image.
- `generated_samples/evaluation_summary.md` — overall and per-class pass rates, confidence, checkpoint metrics, and a short interpretation.

Use the PNG as a qualitative results figure in the report or presentation. Use the CSV for reproducible quantitative analysis, and adapt the Markdown interpretation and summary metrics for the evaluation section and results slides. The generated artifacts remain ignored by Git by default; copy selected final figures into the report or slides workflow only when the group decides how they should be published.

To generate more than eight samples per class or use CPU explicitly:

```bash
uv run python generate_project_outputs.py --num-images-per-class 12 --device cpu
```

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

- Do not commit `data/`; Fashion-MNIST is downloaded locally by the training scripts.
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
