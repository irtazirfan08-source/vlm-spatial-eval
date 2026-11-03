"""Automated end-to-end spatial reasoning benchmark runner and report generator."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image, ImageDraw

from .directional_evaluator import SpatialRelation
from .evaluator import VLMSpatialEvaluator
from .metrics_aggregator import SpatialMetricsAggregator
from .perturbations import PhotometricDegradationEngine
from .visualizer import plot_robustness_curves, render_spatial_failure_overlay


def generate_synthetic_benchmark_dataset(samples_per_relation: int = 2) -> List[Dict[str, Any]]:
    """Generates a synthetic spatial benchmark dataset with geometric shapes and bounding box annotations."""
    dataset: List[Dict[str, Any]] = []
    relations = [
        SpatialRelation.LEFT_OF,
        SpatialRelation.RIGHT_OF,
        SpatialRelation.ABOVE,
        SpatialRelation.BELOW,
        SpatialRelation.INSIDE,
    ]

    for rel in relations:
        for i in range(samples_per_relation):
            sample_id = f"synth_{rel.value}_{i+1}"
            img = Image.new("RGB", (128, 128), color="white")
            draw = ImageDraw.Draw(img)

            if rel == SpatialRelation.LEFT_OF:
                subj_box = (10.0, 44.0, 40.0, 74.0)
                ref_box = (70.0, 44.0, 100.0, 74.0)
                pos_prompt = "a blue square to the left of a red square"
                neg_prompt = "a blue square to the right of a red square"
            elif rel == SpatialRelation.RIGHT_OF:
                subj_box = (70.0, 44.0, 100.0, 74.0)
                ref_box = (10.0, 44.0, 40.0, 74.0)
                pos_prompt = "a blue square to the right of a red square"
                neg_prompt = "a blue square to the left of a red square"
            elif rel == SpatialRelation.ABOVE:
                subj_box = (44.0, 10.0, 74.0, 40.0)
                ref_box = (44.0, 70.0, 74.0, 100.0)
                pos_prompt = "a blue square above a red square"
                neg_prompt = "a blue square below a red square"
            elif rel == SpatialRelation.BELOW:
                subj_box = (44.0, 70.0, 74.0, 100.0)
                ref_box = (44.0, 10.0, 74.0, 40.0)
                pos_prompt = "a blue square below a red square"
                neg_prompt = "a blue square above a red square"
            else:  # INSIDE
                subj_box = (40.0, 40.0, 60.0, 60.0)
                ref_box = (20.0, 20.0, 80.0, 80.0)
                pos_prompt = "a blue square inside a red frame"
                neg_prompt = "a blue square disjoint from a red frame"

            draw.rectangle(ref_box, fill="red" if rel != SpatialRelation.INSIDE else None, outline="red")
            draw.rectangle(subj_box, fill="blue")

            dataset.append({
                "id": sample_id,
                "image": img,
                "positive_prompt": pos_prompt,
                "negative_prompt": neg_prompt,
                "subject_box": subj_box,
                "reference_box": ref_box,
                "target_relation": rel.value,
            })

    return dataset


def run_benchmark_pipeline(
    evaluator: Optional[VLMSpatialEvaluator] = None,
    dataset: Optional[List[Dict[str, Any]]] = None,
    output_dir: str = "reports",
    severities: Optional[List[int]] = None,
    save_reports: bool = True,
    save_plots: bool = False,
) -> Dict[str, Any]:
    """Runs zero-shot spatial evaluation across degradation tiers and exports Markdown, LaTeX, JSON, and plot figures."""
    if severities is None:
        severities = [0, 1, 2, 3]

    if dataset is None:
        dataset = generate_synthetic_benchmark_dataset()

    if evaluator is None:
        evaluator = VLMSpatialEvaluator()

    aggregator = SpatialMetricsAggregator()
    severity_accuracies: List[float] = []
    failure_cases: List[Dict[str, Any]] = []

    for severity in severities:
        is_perturbed = severity > 0
        pert_type = f"photometric_severity_{severity}" if is_perturbed else None
        sev_correct = 0

        for sample in dataset:
            degraded_img = PhotometricDegradationEngine.apply_pipeline(
                sample["image"], severity=severity
            )

            result = evaluator.score_pair(
                degraded_img, sample["positive_prompt"], sample["negative_prompt"]
            )

            is_correct = result["is_correct"]
            predicted_label = sample["target_relation"] if is_correct else "incorrect"

            if is_correct:
                sev_correct += 1
            elif save_plots:
                failure_cases.append({
                    "image": degraded_img,
                    "subject_box": sample["subject_box"],
                    "reference_box": sample["reference_box"],
                    "target_relation": sample["target_relation"],
                    "predicted_relation": "incorrect",
                    "filename": f"fail_{sample['id']}_sev_{severity}.png",
                })

            aggregator.add_evaluation(
                sample_id=f"{sample['id']}_sev_{severity}",
                relation=sample["target_relation"],
                predicted=predicted_label,
                ground_truth=sample["target_relation"],
                is_perturbed=is_perturbed,
                perturbation_type=pert_type,
            )

        acc = (sev_correct / len(dataset) * 100.0) if dataset else 0.0
        severity_accuracies.append(acc)

    reports_generated: Dict[str, str] = {}
    plots_generated: Dict[str, Any] = {}
    out_path = Path(output_dir)

    if save_reports:
        out_path.mkdir(parents=True, exist_ok=True)
        md_file = out_path / "benchmark_summary.md"
        latex_file = out_path / "benchmark_report.tex"
        json_file = out_path / "benchmark_summary.json"

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(aggregator.to_markdown_table())

        aggregator.export_latex(str(latex_file))
        aggregator.export_json(str(json_file))

        reports_generated = {
            "markdown": str(md_file),
            "latex": str(latex_file),
            "json": str(json_file),
        }

    if save_plots:
        figures_dir = out_path / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        curve_file = figures_dir / "robustness_curves.png"
        curve_payload = {
            "VLMSpatialEvaluator": {
                "accuracies": severity_accuracies,
                "color": "#2563eb",
                "marker": "o",
                "linestyle": "-",
            }
        }
        plot_robustness_curves(
            results=curve_payload,
            severities=severities,
            output_path=str(curve_file),
            title="Spatial Reasoning Degradation Across Photometric Severities",
        )
        plots_generated["curve"] = str(curve_file)

        failures_dir = figures_dir / "failures"
        failures_dir.mkdir(parents=True, exist_ok=True)
        rendered_failures: List[str] = []
        for fail in failure_cases[:10]:
            fail_path = failures_dir / fail["filename"]
            render_spatial_failure_overlay(
                image=fail["image"],
                subject_box=fail["subject_box"],
                reference_box=fail["reference_box"],
                target_relation=fail["target_relation"],
                predicted_relation=fail["predicted_relation"],
                output_path=str(fail_path),
            )
            rendered_failures.append(str(fail_path))
        plots_generated["failures"] = rendered_failures

    return {
        "aggregator": aggregator,
        "clean_accuracy": aggregator.get_clean_accuracy(),
        "perturbed_accuracy": aggregator.get_perturbed_accuracy(),
        "robustness_drop": aggregator.compute_robustness_drop(),
        "reports": reports_generated,
        "plots": plots_generated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM Spatial Reasoning Benchmark Suite")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory for exported reports")
    parser.add_argument("--severities", nargs="+", type=int, default=[0, 1, 2, 3], help="Photometric degradation levels")
    parser.add_argument("--samples-per-rel", type=int, default=2, help="Number of synthetic samples per relation")
    parser.add_argument("--device", type=str, default=None, help="Device to run CLIP model ('cpu' or 'cuda')")
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32", help="Pretrained model identifier")
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Export robustness degradation curves and failure diagnostic overlays into reports/figures/",
    )
    args = parser.parse_args()

    print(f"Initializing VLMSpatialEvaluator ({args.model_name})...")
    evaluator = VLMSpatialEvaluator(model_name=args.model_name, device=args.device)

    print("Generating synthetic spatial benchmark dataset...")
    dataset = generate_synthetic_benchmark_dataset(samples_per_relation=args.samples_per_rel)
    print(f"Dataset generated: {len(dataset)} samples across 5 spatial predicates.")

    print(f"Executing benchmark across degradation severities: {args.severities}...")
    results = run_benchmark_pipeline(
        evaluator=evaluator,
        dataset=dataset,
        output_dir=args.output_dir,
        severities=args.severities,
        save_reports=True,
        save_plots=args.save_plots,
    )

    print("\n" + results["aggregator"].generate_markdown_report())
    print(f"\nBenchmark reports successfully saved to: {args.output_dir}/")
    if args.save_plots:
        print(f"Diagnostic plots saved to: {args.output_dir}/figures/")


if __name__ == "__main__":
    main()