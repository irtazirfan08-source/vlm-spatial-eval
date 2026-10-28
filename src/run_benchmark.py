"""Automated end-to-end spatial reasoning benchmark runner and report generator."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image, ImageDraw

from .directional_evaluator import BoundingBox, SpatialRelation
from .evaluator import VLMSpatialEvaluator
from .metrics_aggregator import SpatialMetricsAggregator
from .perturbations import PhotometricDegradationEngine


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
) -> Dict[str, Any]:
    """Runs zero-shot spatial evaluation across degradation tiers and exports Markdown, LaTeX, and JSON reports."""
    if severities is None:
        severities = [0, 1, 2, 3]

    if dataset is None:
        dataset = generate_synthetic_benchmark_dataset()

    if evaluator is None:
        evaluator = VLMSpatialEvaluator()

    aggregator = SpatialMetricsAggregator()

    for severity in severities:
        is_perturbed = severity > 0
        pert_type = f"photometric_severity_{severity}" if is_perturbed else None

        for sample in dataset:
            degraded_img = PhotometricDegradationEngine.apply_pipeline(
                sample["image"], severity=severity
            )

            result = evaluator.score_pair(
                degraded_img, sample["positive_prompt"], sample["negative_prompt"]
            )

            predicted_label = (
                sample["target_relation"] if result["is_correct"] else "incorrect"
            )

            aggregator.add_evaluation(
                sample_id=f"{sample['id']}_sev_{severity}",
                relation=sample["target_relation"],
                predicted=predicted_label,
                ground_truth=sample["target_relation"],
                is_perturbed=is_perturbed,
                perturbation_type=pert_type,
            )

    reports_generated: Dict[str, str] = {}
    if save_reports:
        out_path = Path(output_dir)
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

    return {
        "aggregator": aggregator,
        "clean_accuracy": aggregator.get_clean_accuracy(),
        "perturbed_accuracy": aggregator.get_perturbed_accuracy(),
        "robustness_drop": aggregator.compute_robustness_drop(),
        "reports": reports_generated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM Spatial Reasoning Benchmark Suite")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory for exported reports")
    parser.add_argument("--severities", nargs="+", type=int, default=[0, 1, 2, 3], help="Photometric degradation levels")
    parser.add_argument("--samples-per-rel", type=int, default=2, help="Number of synthetic samples per relation")
    parser.add_argument("--device", type=str, default=None, help="Device to run CLIP model ('cpu' or 'cuda')")
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32", help="Pretrained model identifier")
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
    )

    print("\n" + results["aggregator"].generate_markdown_report())
    print(f"\nBenchmark reports successfully saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()