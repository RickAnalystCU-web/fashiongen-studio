"""Train a small experimental conditional diffusion model on Fashion-MNIST."""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from helper_lib.fashion_data import (
    FASHION_MNIST_CLASSES,
    get_fashion_mnist_loaders,
)
from helper_lib.fashion_diffusion import (
    DiffusionSchedule,
    generate_diffusion_samples,
    get_fashion_diffusion_model,
    q_sample,
)
from helper_lib.fashion_generator import generated_tensor_to_pil
from helper_lib.utils import get_device, save_checkpoint, set_seed


def positive_int(value: str) -> int:
    """Parse and validate a positive integer argparse value."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse diffusion training options."""

    parser = argparse.ArgumentParser(
        description="Train the experimental FashionGen conditional diffusion model."
    )
    parser.add_argument("--epochs", type=positive_int, default=1)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--timesteps", type=positive_int, default=100)
    parser.add_argument("--base-channels", type=positive_int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Custom checkpoint path; defaults to an epoch-specific filename.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-path",
        default=None,
        help="Custom sample-grid path; defaults to an epoch-specific filename.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Custom Markdown path; defaults to an epoch-specific filename.",
    )
    return parser.parse_args()


def resolve_output_paths(
    epochs: int,
    checkpoint_path: str | Path | None = None,
    sample_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve custom paths or create non-overwriting epoch-specific defaults."""

    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    epoch_suffix = f"{epochs}ep"
    resolved_checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else Path("checkpoints") / f"fashion_diffusion_{epoch_suffix}.pth"
    )
    resolved_sample = (
        Path(sample_path)
        if sample_path is not None
        else Path("generated_samples")
        / f"diffusion_samples_grid_{epoch_suffix}.png"
    )
    resolved_summary = (
        Path(summary_path)
        if summary_path is not None
        else Path("generated_samples") / f"diffusion_summary_{epoch_suffix}.md"
    )
    return resolved_checkpoint, resolved_sample, resolved_summary


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ["DejaVuSans-Bold.ttf", "arialbd.ttf"]
        if bold
        else ["DejaVuSans.ttf", "arial.ttf"]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def save_sample_grid(
    images: torch.Tensor,
    output_path: Path,
    epochs: int,
) -> None:
    """Save one labeled row containing all ten Fashion-MNIST classes."""

    cell_width = 128
    image_size = 112
    top_margin = 100
    bottom_margin = 55
    canvas = Image.new(
        "RGB",
        (
            cell_width * len(FASHION_MNIST_CLASSES),
            top_margin + image_size + bottom_margin,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(30, bold=True)
    label_font = _load_font(15, bold=True)
    title = f"FashionGen: {epochs}-Epoch Conditional Diffusion Samples"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    title_x = max((canvas.width - (title_box[2] - title_box[0])) // 2, 0)
    draw.text((title_x, 25), title, fill="#111827", font=title_font)

    for class_index, class_name in enumerate(FASHION_MNIST_CLASSES):
        thumbnail = generated_tensor_to_pil(images[class_index]).resize(
            (image_size, image_size),
            resample=Image.Resampling.NEAREST,
        ).convert("RGB")
        left = class_index * cell_width + (cell_width - image_size) // 2
        canvas.paste(thumbnail, (left, top_margin))
        label = f"{class_index}: {class_name}"
        label_box = draw.textbbox((0, 0), label, font=label_font)
        label_x = class_index * cell_width + max(
            (cell_width - (label_box[2] - label_box[0])) // 2,
            0,
        )
        draw.text(
            (label_x, top_margin + image_size + 14),
            label,
            fill="#1F2937",
            font=label_font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def save_training_summary(
    output_path: Path,
    checkpoint_path: Path,
    sample_path: Path,
    args: argparse.Namespace,
    final_loss: float,
    training_seconds: float,
    device: torch.device,
    peak_gpu_memory_mb: float | None,
) -> None:
    """Write a report-ready description matching the completed epoch count."""

    device_description = (
        torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    )
    memory_line = (
        f"- Peak allocated GPU memory: **{peak_gpu_memory_mb:.1f} MiB**"
        if peak_gpu_memory_mb is not None
        else "- Peak allocated GPU memory: not applicable"
    )
    if args.epochs == 1:
        run_description = "a deliberately limited one-epoch smoke test"
        interpretation = (
            "Only one epoch was used to verify the conditional diffusion pipeline on "
            "GPU. The samples are an early qualitative baseline, not a converged "
            "result. Visible noise, weak silhouettes, or limited class separation "
            "are expected at this stage."
        )
    else:
        run_description = (
            f"a short experimental diffusion training run of {args.epochs} epochs"
        )
        interpretation = (
            f"This {args.epochs}-epoch run explores whether additional training "
            "improves image structure and class conditioning beyond the smoke test. "
            "The samples remain an experimental comparison with the CVAE rather than "
            "a replacement for the productionized generator."
        )

    lines = [
        "# Experimental Fashion-MNIST Diffusion Summary",
        "",
        f"This artifact records {run_description}. The Conditional VAE remains "
        "FashionGen Studio's main productionized generator.",
        "",
        "## Configuration and results",
        "",
        f"- Epochs: **{args.epochs}**",
        f"- Batch size: **{args.batch_size}**",
        f"- Diffusion timesteps: **{args.timesteps}**",
        f"- Base channels: **{args.base_channels}**",
        "- Noise schedule: **cosine**",
        "- Objective: **epsilon prediction with mean squared error**",
        f"- Device: **{device} ({device_description})**",
        f"- Training time: **{training_seconds:.1f} seconds**",
        f"- Final training loss: **{final_loss:.6f}**",
        memory_line,
        f"- Checkpoint: `{checkpoint_path.as_posix()}`",
        f"- Sample grid: `{sample_path.as_posix()}`",
        "",
        "## Interpretation",
        "",
        interpretation,
        "Longer training should be considered only if this stretch goal adds value "
        "beyond the completed CVAE workflow.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Train for the requested short run, save weights, and create artifacts."""

    args = parse_args()
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.timesteps <= 1:
        raise ValueError("--timesteps must be greater than one.")
    if args.base_channels < 8 or args.base_channels % 8 != 0:
        raise ValueError("--base-channels must be at least 8 and divisible by 8.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")

    checkpoint_path, sample_path, summary_path = resolve_output_paths(
        epochs=args.epochs,
        checkpoint_path=args.checkpoint_path,
        sample_path=args.sample_path,
        summary_path=args.summary_path,
    )

    set_seed(args.seed)
    device = get_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, _, _ = get_fashion_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        download=True,
    )
    model = get_fashion_diffusion_model(
        base_channels=args.base_channels,
        num_classes=len(FASHION_MNIST_CLASSES),
    ).to(device)
    schedule = DiffusionSchedule(args.timesteps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "epoch_seconds": [],
    }

    training_start = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_images = 0
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            timesteps = torch.randint(
                0,
                args.timesteps,
                (images.size(0),),
                device=device,
            )
            noisy_images, noise = q_sample(images, timesteps, schedule)
            optimizer.zero_grad(set_to_none=True)
            predicted_noise = model(noisy_images, timesteps, labels)
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_images += images.size(0)
            if batch_index == 1 or batch_index % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} "
                    f"batch {batch_index}/{len(train_loader)} "
                    f"loss={loss.item():.6f}"
                )

        epoch_loss = total_loss / total_images
        epoch_seconds = time.perf_counter() - epoch_start
        history["train_loss"].append(epoch_loss)
        history["epoch_seconds"].append(epoch_seconds)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={epoch_loss:.6f}, time={epoch_seconds:.1f}s"
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_start
    final_loss = history["train_loss"][-1]
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else None
    )

    training_args = vars(args).copy()
    training_args.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "sample_path": str(sample_path),
            "summary_path": str(summary_path),
        }
    )
    checkpoint_path = save_checkpoint(
        {
            "model_state_dict": model.state_dict(),
            "class_names": list(FASHION_MNIST_CLASSES),
            "timesteps": args.timesteps,
            "base_channels": args.base_channels,
            "schedule": "cosine",
            "training_history": history,
            "training_args": training_args,
            "training_seconds": training_seconds,
            "final_train_loss": final_loss,
            "device": str(device),
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
        },
        checkpoint_path,
    )

    samples = generate_diffusion_samples(
        model=model,
        class_labels=list(range(len(FASHION_MNIST_CLASSES))),
        schedule=schedule,
        device=device,
        seed=args.seed,
    )
    save_sample_grid(samples, sample_path, epochs=args.epochs)
    save_training_summary(
        output_path=summary_path,
        checkpoint_path=Path(checkpoint_path),
        sample_path=sample_path,
        args=args,
        final_loss=final_loss,
        training_seconds=training_seconds,
        device=device,
        peak_gpu_memory_mb=peak_gpu_memory_mb,
    )

    print(f"Training time: {training_seconds:.1f} seconds")
    print(f"Final train loss: {final_loss:.6f}")
    if peak_gpu_memory_mb is not None:
        print(f"Peak allocated GPU memory: {peak_gpu_memory_mb:.1f} MiB")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Sample grid saved to: {sample_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
