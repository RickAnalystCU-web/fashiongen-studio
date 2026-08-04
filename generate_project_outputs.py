"""Generate report and slide artifacts from the trained FashionGen models."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

from helper_lib.fashion_data import FASHION_MNIST_CLASSES
from helper_lib.fashion_generator import (
    classify_fashion_tensors,
    generated_tensor_to_pil,
    get_cached_fashion_classifier,
    get_cached_fashion_cvae,
    sample_fashion_images,
)
from helper_lib.utils import get_device, set_seed


DEFAULT_CVAE_CHECKPOINT = Path("checkpoints/fashion_cvae.pth")
DEFAULT_CLASSIFIER_CHECKPOINT = Path("checkpoints/fashion_classifier.pth")
DEFAULT_OUTPUT_DIR = Path("generated_samples")
CSV_FIELDS = [
    "requested_class_index",
    "requested_label",
    "sample_id",
    "predicted_class_index",
    "predicted_label",
    "confidence",
    "passed_quality_check",
]


def at_least_eight(value: str) -> int:
    """Argparse validator enforcing the project artifact sample minimum."""

    parsed = int(value)
    if parsed < 8:
        raise argparse.ArgumentTypeError("value must be at least 8")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse artifact-generation options."""

    parser = argparse.ArgumentParser(
        description="Generate FashionGen grids and quality-check summaries."
    )
    parser.add_argument(
        "--num-images-per-class",
        type=at_least_eight,
        default=8,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cvae-checkpoint", default=str(DEFAULT_CVAE_CHECKPOINT))
    parser.add_argument(
        "--classifier-checkpoint",
        default=str(DEFAULT_CLASSIFIER_CHECKPOINT),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a portable TrueType font with a Pillow default fallback."""

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


def centered_text_x(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    left: int,
    width: int,
) -> int:
    """Return an x-coordinate that centers text in a horizontal region."""

    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    return left + max((width - text_width) // 2, 0)


def save_quality_grid(
    images: torch.Tensor,
    rows: list[dict[str, Any]],
    per_class_stats: dict[int, dict[str, float | int]],
    images_per_class: int,
    output_path: Path,
) -> None:
    """Save a labeled ten-row image grid with classifier agreement borders."""

    label_width = 340
    cell_width = 165
    cell_height = 180
    thumbnail_size = 140
    top_margin = 165
    right_margin = 20
    bottom_margin = 20
    canvas_width = label_width + images_per_class * cell_width + right_margin
    canvas_height = top_margin + len(FASHION_MNIST_CLASSES) * cell_height + bottom_margin

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(42, bold=True)
    subtitle_font = load_font(22)
    header_font = load_font(20, bold=True)
    row_font = load_font(25, bold=True)
    rate_font = load_font(21)
    prediction_font = load_font(16)

    title = "FashionGen Studio: Conditional VAE Samples"
    draw.text(
        (centered_text_x(draw, title, title_font, 0, canvas_width), 18),
        title,
        fill="#111827",
        font=title_font,
    )
    subtitle = "Green border = CNN matches requested class; red border = mismatch"
    draw.text(
        (centered_text_x(draw, subtitle, subtitle_font, 0, canvas_width), 75),
        subtitle,
        fill="#4B5563",
        font=subtitle_font,
    )

    for column in range(images_per_class):
        label = f"Sample {column + 1}"
        cell_left = label_width + column * cell_width
        draw.text(
            (
                centered_text_x(draw, label, header_font, cell_left, cell_width),
                127,
            ),
            label,
            fill="#374151",
            font=header_font,
        )

    for class_index, class_name in enumerate(FASHION_MNIST_CLASSES):
        row_top = top_margin + class_index * cell_height
        stats = per_class_stats[class_index]
        row_label = f"{class_index}: {class_name}"
        pass_label = (
            f"Pass rate: {int(stats['passed'])}/{int(stats['total'])} "
            f"({float(stats['pass_rate']):.1%})"
        )
        draw.text((20, row_top + 44), row_label, fill="#111827", font=row_font)
        draw.text((20, row_top + 82), pass_label, fill="#4B5563", font=rate_font)

        for column in range(images_per_class):
            flat_index = class_index * images_per_class + column
            record = rows[flat_index]
            thumbnail = generated_tensor_to_pil(images[flat_index]).resize(
                (thumbnail_size, thumbnail_size),
                resample=Image.Resampling.NEAREST,
            ).convert("RGB")
            cell_left = label_width + column * cell_width
            image_left = cell_left + (cell_width - thumbnail_size) // 2
            image_top = row_top + 2
            canvas.paste(thumbnail, (image_left, image_top))

            passed = bool(record["passed_quality_check"])
            border_color = "#16A34A" if passed else "#DC2626"
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
            prediction_text = (
                f"{record['predicted_label']} "
                f"{float(record['confidence']):.2f}"
            )
            draw.text(
                (
                    centered_text_x(
                        draw,
                        prediction_text,
                        prediction_font,
                        cell_left,
                        cell_width,
                    ),
                    row_top + 149,
                ),
                prediction_text,
                fill="#1F2937",
                font=prediction_font,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def save_csv_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write one classifier result row per generated image."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "confidence": f"{float(row['confidence']):.6f}",
                }
            )


def metadata_metric(
    metadata: dict[str, Any],
    section: str,
    metric: str,
) -> float | None:
    """Read an optional numeric metric from checkpoint metadata."""

    value = metadata.get(section, {}).get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def save_markdown_summary(
    rows: list[dict[str, Any]],
    per_class_stats: dict[int, dict[str, float | int]],
    cvae_checkpoint: Path,
    classifier_checkpoint: Path,
    cvae_metadata: dict[str, Any],
    classifier_metadata: dict[str, Any],
    seed: int,
    output_path: Path,
) -> None:
    """Write quantitative results and a short report-ready interpretation."""

    total_images = len(rows)
    total_passed = sum(bool(row["passed_quality_check"]) for row in rows)
    overall_pass_rate = total_passed / total_images
    average_confidence = sum(float(row["confidence"]) for row in rows) / total_images
    classifier_accuracy = metadata_metric(
        classifier_metadata,
        "test_metrics",
        "accuracy",
    )
    cvae_validation_loss = metadata_metric(
        cvae_metadata,
        "validation_metrics",
        "loss",
    )

    ordered_classes = sorted(
        per_class_stats.items(),
        key=lambda item: float(item[1]["pass_rate"]),
    )
    weakest_rate = float(ordered_classes[0][1]["pass_rate"])
    weakest_classes = [
        FASHION_MNIST_CLASSES[index]
        for index, stats in ordered_classes
        if float(stats["pass_rate"]) == weakest_rate
    ]
    strongest_index, strongest_stats = ordered_classes[-1]

    lines = [
        "# FashionGen Studio Evaluation Summary",
        "",
        "## Run configuration",
        "",
        f"- Seed: `{seed}`",
        f"- Total generated images: **{total_images}**",
        f"- CVAE checkpoint: `{cvae_checkpoint.as_posix()}`",
        f"- Classifier checkpoint: `{classifier_checkpoint.as_posix()}`",
        "",
        "## Overall results",
        "",
        f"- Passed quality checks: **{total_passed}/{total_images}**",
        f"- Overall pass rate: **{overall_pass_rate:.1%}**",
        f"- Average classifier confidence: **{average_confidence:.3f}**",
        (
            f"- Classifier checkpoint test accuracy: **{classifier_accuracy:.2f}%**"
            if classifier_accuracy is not None
            else "- Classifier checkpoint test accuracy: not available"
        ),
        (
            f"- CVAE checkpoint validation loss: **{cvae_validation_loss:.4f}**"
            if cvae_validation_loss is not None
            else "- CVAE checkpoint validation loss: not available"
        ),
        "",
        "## Per-class quality-check results",
        "",
        "| Class index | Requested class | Passed | Total | Pass rate | Average confidence |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for class_index, class_name in enumerate(FASHION_MNIST_CLASSES):
        stats = per_class_stats[class_index]
        lines.append(
            f"| {class_index} | {class_name} | {int(stats['passed'])} | "
            f"{int(stats['total'])} | {float(stats['pass_rate']):.1%} | "
            f"{float(stats['average_confidence']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation for report and slides",
            "",
            (
                f"The CNN agreed with the requested class for {overall_pass_rate:.1%} "
                f"of CVAE samples, with average confidence {average_confidence:.3f}. "
                "This agreement rate is a practical automated quality proxy: it "
                "measures whether generated class features are recognizable to an "
                "independently trained Fashion-MNIST classifier."
            ),
            "",
            (
                f"The strongest row was **{FASHION_MNIST_CLASSES[strongest_index]}** "
                f"at {float(strongest_stats['pass_rate']):.1%}. The weakest "
                f"class(es) were **{', '.join(weakest_classes)}** at "
                f"{weakest_rate:.1%}. Lower-performing classes likely reflect "
                "visual overlap among Fashion-MNIST clothing categories or less "
                "distinct CVAE samples and should be highlighted as improvement areas."
            ),
            "",
            (
                "The classifier agreement score is not a substitute for human visual "
                "assessment. The accompanying grid should be used to discuss image "
                "clarity, diversity, and recognizable class structure in the final "
                "report and presentation."
            ),
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_per_class_stats(
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, float | int]]:
    """Aggregate generated-image classifier results by requested class."""

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["requested_class_index"])].append(row)

    stats: dict[int, dict[str, float | int]] = {}
    for class_index in range(len(FASHION_MNIST_CLASSES)):
        class_rows = grouped[class_index]
        passed = sum(bool(row["passed_quality_check"]) for row in class_rows)
        total = len(class_rows)
        stats[class_index] = {
            "passed": passed,
            "total": total,
            "pass_rate": passed / total,
            "average_confidence": sum(
                float(row["confidence"]) for row in class_rows
            )
            / total,
        }
    return stats


def main() -> None:
    """Generate the grid, CSV results, and Markdown evaluation summary."""

    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    cvae_checkpoint_path = Path(args.cvae_checkpoint)
    classifier_checkpoint_path = Path(args.classifier_checkpoint)
    output_dir = Path(args.output_dir)

    cvae, cvae_metadata, cvae_device = get_cached_fashion_cvae(
        cvae_checkpoint_path,
        device=device,
    )
    classifier, classifier_metadata, classifier_device = (
        get_cached_fashion_classifier(
            classifier_checkpoint_path,
            device=device,
        )
    )
    class_names = list(
        classifier_metadata.get("class_names", FASHION_MNIST_CLASSES)
    )
    if class_names != list(FASHION_MNIST_CLASSES):
        raise ValueError("Classifier checkpoint class order does not match Fashion-MNIST.")

    requested_labels = [
        class_index
        for class_index in range(len(FASHION_MNIST_CLASSES))
        for _ in range(args.num_images_per_class)
    ]
    images = sample_fashion_images(
        model=cvae,
        class_labels=requested_labels,
        device=cvae_device,
        seed=args.seed,
    )
    predictions = classify_fashion_tensors(
        model=classifier,
        images=images,
        class_names=class_names,
        device=classifier_device,
        top_k=3,
    )

    rows: list[dict[str, Any]] = []
    for flat_index, (requested_index, prediction) in enumerate(
        zip(requested_labels, predictions, strict=True)
    ):
        sample_id = flat_index % args.num_images_per_class + 1
        predicted_index = int(prediction["predicted_class_index"])
        rows.append(
            {
                "requested_class_index": requested_index,
                "requested_label": FASHION_MNIST_CLASSES[requested_index],
                "sample_id": sample_id,
                "predicted_class_index": predicted_index,
                "predicted_label": prediction["predicted_label"],
                "confidence": float(prediction["confidence"]),
                "passed_quality_check": predicted_index == requested_index,
            }
        )

    per_class_stats = build_per_class_stats(rows)
    grid_path = output_dir / "cvae_all_classes_grid.png"
    csv_path = output_dir / "generate_and_check_summary.csv"
    markdown_path = output_dir / "evaluation_summary.md"

    save_quality_grid(
        images=images,
        rows=rows,
        per_class_stats=per_class_stats,
        images_per_class=args.num_images_per_class,
        output_path=grid_path,
    )
    save_csv_summary(rows, csv_path)
    save_markdown_summary(
        rows=rows,
        per_class_stats=per_class_stats,
        cvae_checkpoint=cvae_checkpoint_path,
        classifier_checkpoint=classifier_checkpoint_path,
        cvae_metadata=cvae_metadata,
        classifier_metadata=classifier_metadata,
        seed=args.seed,
        output_path=markdown_path,
    )

    total_passed = sum(bool(row["passed_quality_check"]) for row in rows)
    overall_pass_rate = total_passed / len(rows)
    print(f"Device: {device}")
    print(f"Generated images: {len(rows)}")
    print(f"Overall pass rate: {total_passed}/{len(rows)} ({overall_pass_rate:.1%})")
    print("Per-class pass rates:")
    for class_index, class_name in enumerate(FASHION_MNIST_CLASSES):
        stats = per_class_stats[class_index]
        print(
            f"  {class_index}: {class_name:<12} "
            f"{int(stats['passed'])}/{int(stats['total'])} "
            f"({float(stats['pass_rate']):.1%})"
        )
    print(f"Saved grid: {grid_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
