import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be resolved directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from src.evaluator import VLMSpatialEvaluator
from src.spatial_generator import SpatialDatasetGenerator


def run_benchmark():
    print("\n--- Initializing VLM Spatial Robustness Benchmark ---")
    generator = SpatialDatasetGenerator(canvas_size=224)
    dataset = generator.generate_benchmark_split(num_samples_per_relation=8)
    print(f"Generated evaluation set: {len(dataset)} balanced spatial pairs.")

    evaluator = VLMSpatialEvaluator(model_name="openai/clip-vit-base-patch32")

    severities = [0, 1, 2, 3, 4]
    accuracies = []
    margins = []

    print("\n| Severity Level | Accuracy (%) | Mean Margin |")
    print("|----------------|--------------|-------------|")

    for s in severities:
        metrics = evaluator.evaluate_dataset_at_severity(dataset, severity=s)
        accuracies.append(metrics["accuracy"])
        margins.append(metrics["mean_margin"])
        print(f"| L_{s:<12} | {metrics['accuracy']:<12.1f} | {metrics['mean_margin']:<11.4f} |")

    # Generate degradation curve plot
    plt.figure(figsize=(8, 4.5))
    plt.plot(
        severities,
        accuracies,
        marker="o",
        color="#2563eb",
        linewidth=2.5,
        label="CLIP ViT-B/32",
    )
    plt.title("Spatial Relational Reasoning vs. Photometric Degradation", fontsize=12)
    plt.xlabel("Severity Level (L0=Clean, L4=Deep Underexposure + Sensor Noise)", fontsize=10)
    plt.ylabel("Relational Pair Accuracy (%)", fontsize=10)
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    output_path = "perception_collapse_curve.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nBenchmark completed successfully. Saved degradation curve to '{output_path}'.")


if __name__ == "__main__":
    run_benchmark()