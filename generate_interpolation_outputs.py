"""Generate slide-ready Conditional VAE latent interpolation artifacts."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from helper_lib.fashion_data import FASHION_MNIST_CLASSES
from helper_lib.fashion_generator import (
    generated_tensor_to_pil,
    get_cached_fashion_cvae,
    sample_fashion_latent_interpolation,
)
from helper_lib.utils import get_device, set_seed


DEFAULT_CVAE_CHECKPOINT = Path("checkpoints/fashion_cvae.pth")
DEFAULT_OUTPUT_DIR = Path("generated_samples")
INTERPOLATION_EXAMPLES = [
    (7, 9),  # Sneaker -> Ankle boot
    (0, 2),  # T-shirt/top -> Pullover
    (8, 7),  # Bag -> Sneaker
]


def interpolation_steps(value: str) -> int:
    """Validate a readable interpolation-grid step count."""

    parsed = int(value)
    if not 4 <= parsed <= 16:
        raise argparse.ArgumentTypeError("steps must be between 4 and 16")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse interpolation artifact options."""

    parser = argparse.ArgumentParser(
        description="Generate FashionGen latent interpolation outputs."
    )
    parser.add_argument("--steps", type=interpolation_steps, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cvae-checkpoint", default=str(DEFAULT_CVAE_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a portable TrueType font with a Pillow fallback."""

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
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def centered_x(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    left: int,
    width: int,
) -> int:
    """Center text horizontally within a region."""

    bounds = draw.textbbox((0, 0), text, font=font)
    return left + max((width - (bounds[2] - bounds[0])) // 2, 0)


def save_interpolation_grid(
    interpolation_rows: list[dict[str, object]],
    steps: int,
    output_path: Path,
) -> None:
    """Save a labeled three-row latent interpolation grid."""

    label_width = 350
    cell_width = 170
    cell_height = 190
    thumbnail_size = 145
    top_margin = 170
    right_margin = 20
    bottom_margin = 25
    width = label_width + steps * cell_width + right_margin
    height = top_margin + len(interpolation_rows) * cell_height + bottom_margin

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(40, bold=True)
    subtitle_font = load_font(21)
    header_font = load_font(19, bold=True)
    row_font = load_font(25, bold=True)
    detail_font = load_font(18)
    caption_font = load_font(16)

    title = "FashionGen Studio: CVAE Latent Interpolation"
    draw.text(
        (centered_x(draw, title, title_font, 0, width), 18),
        title,
        fill="#111827",
        font=title_font,
    )
    subtitle = (
        "Linear interpolation in z; blue = start-label condition, "
        "purple = end-label condition"
    )
    draw.text(
        (centered_x(draw, subtitle, subtitle_font, 0, width), 72),
        subtitle,
        fill="#4B5563",
        font=subtitle_font,
    )

    for column in range(steps):
        alpha = column / (steps - 1)
        heading = f"alpha={alpha:.2f}"
        left = label_width + column * cell_width
        draw.text(
            (centered_x(draw, heading, header_font, left, cell_width), 128),
            heading,
            fill="#374151",
            font=header_font,
        )

    for row_index, row in enumerate(interpolation_rows):
        row_top = top_margin + row_index * cell_height
        start_index = int(row["start_index"])
        end_index = int(row["end_index"])
        seed = int(row["seed"])
        images = row["images"]
        label_schedule = row["label_schedule"]
        if not isinstance(images, torch.Tensor) or not isinstance(
            label_schedule, list
        ):
            raise TypeError("Interpolation row contains invalid image data.")

        row_title = (
            f"{FASHION_MNIST_CLASSES[start_index]} -> "
            f"{FASHION_MNIST_CLASSES[end_index]}"
        )
        draw.text((18, row_top + 50), row_title, fill="#111827", font=row_font)
        draw.text(
            (18, row_top + 91),
            f"Seed {seed} | {steps} steps",
            fill="#4B5563",
            font=detail_font,
        )

        for column in range(steps):
            condition_index = int(label_schedule[column])
            condition_name = FASHION_MNIST_CLASSES[condition_index]
            thumbnail = generated_tensor_to_pil(images[column]).resize(
                (thumbnail_size, thumbnail_size),
                resample=Image.Resampling.NEAREST,
            ).convert("RGB")
            cell_left = label_width + column * cell_width
            image_left = cell_left + (cell_width - thumbnail_size) // 2
            image_top = row_top + 2
            canvas.paste(thumbnail, (image_left, image_top))

            border_color = "#2563EB" if condition_index == start_index else "#7C3AED"
            draw.rectangle(
                (
                    image_left - 3,
                    image_top - 3,
                    image_left + thumbnail_size + 2,
                    image_top + thumbnail_size + 2,
                ),
                outline=border_color,
                width=5,
            )
            caption = f"cond: {condition_name}"
            draw.text(
                (
                    centered_x(
                        draw,
                        caption,
                        caption_font,
                        cell_left,
                        cell_width,
                    ),
                    row_top + 153,
                ),
                caption,
                fill="#374151",
                font=caption_font,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def save_interpolation_summary(
    rows: list[dict[str, object]],
    steps: int,
    checkpoint_path: Path,
    output_path: Path,
) -> None:
    """Write a short report-ready explanation of the interpolation artifact."""

    lines = [
        "# FashionGen Studio Latent Interpolation",
        "",
        "## Configuration",
        "",
        f"- CVAE checkpoint: `{checkpoint_path.as_posix()}`",
        f"- Interpolation steps per example: **{steps}**",
        "- Interpolation rule: `z(alpha) = (1 - alpha) * z_start + alpha * z_end`",
        "- Class conditioning: start label for the first half, end label for the second half",
        "",
        "## Examples",
        "",
        "| Start class | End class | Seed | Steps |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        start_index = int(row["start_index"])
        end_index = int(row["end_index"])
        lines.append(
            f"| {FASHION_MNIST_CLASSES[start_index]} | "
            f"{FASHION_MNIST_CLASSES[end_index]} | {int(row['seed'])} | {steps} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The grid demonstrates that nearby points along a straight path in "
                "the learned latent space decode into visually related Fashion-MNIST "
                "images. Gradual shape and intensity changes indicate that the CVAE "
                "learned a structured, continuous representation rather than simply "
                "memorizing isolated training examples."
            ),
            "",
            (
                "Because FashionGen uses discrete class embeddings, the conditioning "
                "label switches at the midpoint of each row. The latent vector itself "
                "changes continuously, but the midpoint may show a sharper semantic "
                "transition. This is a low-risk exploration feature, not evidence of "
                "fully continuous label interpolation."
            ),
            "",
            (
                "For the report or presentation, use the grid to explain latent-space "
                "continuity, class conditioning, and the distinction between smooth "
                "latent movement and discrete semantic control."
            ),
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Generate the three required interpolation examples and documentation."""

    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    checkpoint_path = Path(args.cvae_checkpoint)
    output_dir = Path(args.output_dir)

    model, _, resolved_device = get_cached_fashion_cvae(
        checkpoint_path,
        device=device,
    )
    rows: list[dict[str, object]] = []
    for row_index, (start_index, end_index) in enumerate(INTERPOLATION_EXAMPLES):
        row_seed = args.seed + row_index
        images, label_schedule, alphas = sample_fashion_latent_interpolation(
            model=model,
            start_label=start_index,
            end_label=end_index,
            steps=args.steps,
            device=resolved_device,
            seed=row_seed,
        )
        rows.append(
            {
                "start_index": start_index,
                "end_index": end_index,
                "seed": row_seed,
                "images": images,
                "label_schedule": label_schedule,
                "alphas": alphas,
            }
        )

    grid_path = output_dir / "cvae_interpolation_grid.png"
    summary_path = output_dir / "interpolation_summary.md"
    save_interpolation_grid(rows, args.steps, grid_path)
    save_interpolation_summary(
        rows=rows,
        steps=args.steps,
        checkpoint_path=checkpoint_path,
        output_path=summary_path,
    )

    print(f"Device: {resolved_device}")
    print(f"Interpolation examples: {len(rows)}")
    for row in rows:
        start_index = int(row["start_index"])
        end_index = int(row["end_index"])
        print(
            f"  {FASHION_MNIST_CLASSES[start_index]} -> "
            f"{FASHION_MNIST_CLASSES[end_index]} "
            f"(seed={int(row['seed'])}, steps={args.steps})"
        )
    print(f"Saved grid: {grid_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
