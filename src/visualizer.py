import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from src.evaluator import VLMSpatialEvaluator
from src.autoregressive_evaluator import AutoregressiveSpatialEvaluator
from src.spatial_generator import SpatialDatasetGenerator


def run_benchmark():
    print("\n" + "=" * 65)
    print("   VLM-SpatialEval: Multi-Architecture Robustness Benchmark")
    print("   Evaluating: Contrastive (CLIP, SigLIP) vs. Autoregressive (SmolVLM)")
    print("=" * 65 + "\n")

    # Generate synthetic balanced test split (2 samples per relation = 8 total test pairs)
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

    # Render Multi-Architecture Collapse Plot
    plt.figure(figsize=(9, 5.2))
    for name, data in results.items():
        plt.plot(
            severities,
            data["accuracies"],
            marker=data["marker"],
            color=data["color"],
            linestyle=data["linestyle"],
            linewidth=2.2,
            markersize=6,
            label=name,
        )

    # 50% Random Chance Line
    plt.axhline(y=50.0, color="#6b7280", linestyle=":", linewidth=1.5, label="Random Baseline (50%)")

    plt.title("Spatial Relational Reasoning Across VLM Architectures", fontsize=12, fontweight="bold")
    plt.xlabel("Photometric Degradation Severity (L0=Clean, L4=Severe Noise/Underexposure)", fontsize=10)
    plt.ylabel("Relational Accuracy (%)", fontsize=10)
    plt.ylim(0, 105)
    plt.xticks(severities, [f"L_{s}" for s in severities])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()

    output_path = "perception_collapse_curve.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nBenchmark completed successfully. Updated comparison plot saved to '{output_path}'.")


if __name__ == "__main__":
    run_benchmark()