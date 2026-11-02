"""Visual diagnostic utilities and degradation robustness curve plotting for VLMs."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image, ImageDraw

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluator import VLMSpatialEvaluator
from .autoregressive_evaluator import AutoregressiveSpatialEvaluator
from .spatial_generator import SpatialDatasetGenerator


def _to_box_coords(box: Any) -> Tuple[float, float, float, float]:
    """Extracts (xmin, ymin, xmax, ymax) from either a BoundingBox dataclass or tuple."""
    if hasattr(box, "xmin") and hasattr(box, "ymin") and hasattr(box, "xmax") and hasattr(box, "ymax"):
        return (float(box.xmin), float(box.ymin), float(box.xmax), float(box.ymax))
    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))


def plot_robustness_curves(
    results: Dict[str, Dict[str, Any]],
    severities: List[int],
    output_path: str = "perception_collapse_curve.png",
    title: str = "Spatial Relational Reasoning Across VLM Architectures",
) -> str:
    """Renders and exports multi-model or per-relation robustness degradation curves."""
    fig, ax = plt.subplots(figsize=(9, 5.2))

    for name, data in results.items():
        ax.plot(
            severities,
            data["accuracies"],
            marker=data.get("marker", "o"),
            color=data.get("color", "#2563eb"),
            linestyle=data.get("linestyle", "-"),
            linewidth=data.get("linewidth", 2.2),
            markersize=data.get("markersize", 6),
            label=name,
        )

    # 50% Random Chance Line
    ax.axhline(y=50.0, color="#6b7280", linestyle=":", linewidth=1.5, label="Random Baseline (50%)")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Photometric Degradation Severity (L0=Clean, L4=Severe Noise/Underexposure)", fontsize=10)
    ax.set_ylabel("Relational Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_xticks(severities)
    ax.set_xticklabels([f"L_{s}" for s in severities])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

    return str(out_file)


def render_spatial_failure_overlay(
    image: Image.Image,
    subject_box: Union[Tuple[float, float, float, float], Any],
    reference_box: Union[Tuple[float, float, float, float], Any],
    target_relation: str,
    predicted_relation: str,
    output_path: Optional[str] = None,
) -> Image.Image:
    """Renders a diagnostic failure overlay showing subject/reference boxes and relation mismatch."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    sub_coords = _to_box_coords(subject_box)
    ref_coords = _to_box_coords(reference_box)

    # Reference entity box (red outline)
    draw.rectangle(ref_coords, outline="#dc2626", width=2)
    # Subject entity box (blue outline)
    draw.rectangle(sub_coords, outline="#2563eb", width=2)

    # Status banner at top left
    banner_text = f"GT: {target_relation} | Pred: {predicted_relation}"
    is_correct = str(target_relation).strip().lower() == str(predicted_relation).strip().lower()
    text_color = "#22c55e" if is_correct else "#ef4444"

    draw.rectangle([0, 0, min(canvas.width, 220), 18], fill="#111827")
    draw.text((4, 2), banner_text, fill=text_color)

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_file)

    return canvas


def run_benchmark():
    print("\n" + "=" * 65)
    print("   VLM-SpatialEval: Multi-Architecture Robustness Benchmark")
    print("   Evaluating: Contrastive (CLIP, SigLIP) vs. Autoregressive (SmolVLM)")
    print("=" * 65 + "\n")

    generator = SpatialDatasetGenerator(canvas_size=224)
    dataset = generator.generate_benchmark_split(num_samples_per_relation=2)
    print(f"Synthesized balanced test split: {len(dataset)} spatial pairs across 4 canonical relations.\n")

    severities = [0, 1, 2, 3, 4]
    results = {}

    # 1. Dual-Encoder Contrastive Models
    dual_encoders = [
        {"name": "CLIP ViT-B/32", "id": "openai/clip-vit-base-patch32", "color": "#2563eb", "marker": "o"},
        {"name": "SigLIP Base Patch16", "id": "google/siglip-base-patch16-224", "color": "#dc2626", "marker": "s"},
    ]

    for cfg in dual_encoders:
        print(f"--> Initializing Evaluator: {cfg['name']} ({cfg['id']})")
        evaluator = VLMSpatialEvaluator(model_name=cfg["id"])
        accuracies = []

        for s in severities:
            metrics = evaluator.evaluate_dataset_at_severity(dataset, severity=s)
            accuracies.append(metrics["accuracy"])

        results[cfg["name"]] = {
            "accuracies": accuracies,
            "color": cfg["color"],
            "marker": cfg["marker"],
            "linestyle": "-",
        }
        print(f"    Finished {cfg['name']}: {accuracies}")

    # 2. Autoregressive Vision-Language Model
    print(f"\n--> Initializing Evaluator: SmolVLM-256M-Instruct (Autoregressive Decoder)")
    ar_evaluator = AutoregressiveSpatialEvaluator(model_name="HuggingFaceTB/SmolVLM-256M-Instruct")
    ar_accuracies = []

    for s in severities:
        metrics = ar_evaluator.evaluate_dataset_at_severity(dataset, severity=s)
        ar_accuracies.append(metrics["accuracy"])
        print(f"    Severity L_{s} -> Accuracy: {metrics['accuracy']:.1f}%")

    results["SmolVLM-256M (Generative)"] = {
        "accuracies": ar_accuracies,
        "color": "#16a34a",
        "marker": "^",
        "linestyle": "--",
    }

    # Print Summary Markdown Table
    print("\n" + "=" * 70)
    print("| Severity | CLIP ViT-B/32 (%) | SigLIP Base (%) | SmolVLM-256M (%) |")
    print("|:---------|:------------------|:----------------|:-----------------|")
    for idx, s in enumerate(severities):
        c_acc = results["CLIP ViT-B/32"]["accuracies"][idx]
        sig_acc = results["SigLIP Base Patch16"]["accuracies"][idx]
        smol_acc = results["SmolVLM-256M (Generative)"]["accuracies"][idx]
        print(f"| L_{s:<6} | {c_acc:<17.1f} | {sig_acc:<15.1f} | {smol_acc:<16.1f} |")
    print("=" * 70 + "\n")

    output_path = "perception_collapse_curve.png"
    plot_robustness_curves(results, severities, output_path=output_path)
    print(f"\nBenchmark completed successfully. Updated comparison plot saved to '{output_path}'.")


if __name__ == "__main__":
    run_benchmark()