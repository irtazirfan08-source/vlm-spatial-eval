"""Automated end-to-end spatial reasoning benchmark runner and report generator."""

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from PIL import Image, ImageDraw

from .directional_evaluator import SpatialRelation
from .evaluator import VLMSpatialEvaluator
from .autoregressive_evaluator import AutoregressiveSpatialEvaluator
from .metrics_aggregator import SpatialMetricsAggregator
from .perturbations import PhotometricDegradationEngine
from .visualizer import plot_robustness_curves, render_spatial_failure_overlay
from .counterfactual_evaluator import (
    CounterfactualFlipPair,
    CounterfactualSymmetryAuditor,
    OcclusionSceneRenderer,
    generate_counterfactual_flip_pair,
)
from .calibration import CalibrationTracker, margin_to_confidence


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
                "labels": {
                    "obj_a": ("blue", "square"),
                    "obj_b": ("red", "square" if rel != SpatialRelation.INSIDE else "frame"),
                },
            })

    return dataset


def _extract_prediction_confidence(raw_result: Any) -> float:
    """Resolve top-1 prediction confidence in [0.0, 1.0] from evaluator output or fallback."""
    if isinstance(raw_result, dict):
        if "confidence" in raw_result and raw_result["confidence"] is not None:
            return max(0.0, min(1.0, float(raw_result["confidence"])))
        if "margin" in raw_result and raw_result["margin"] is not None:
            return margin_to_confidence(float(raw_result["margin"]))
        if "pos_score" in raw_result and "neg_score" in raw_result:
            diff = float(raw_result["pos_score"]) - float(raw_result["neg_score"])
            return margin_to_confidence(diff)
    if isinstance(raw_result, (int, float)) and not isinstance(raw_result, bool):
        return margin_to_confidence(float(raw_result))
    return 0.75


def run_benchmark_pipeline(
    evaluator: Any = None,
    dataset: Optional[List[Dict[str, Any]]] = None,
    output_dir: str = "reports",
    severities: Optional[List[int]] = None,
    save_reports: bool = True,
    save_plots: bool = False,
    is_autoregressive: bool = False,
) -> Dict[str, Any]:
    """Runs zero-shot spatial evaluation across degradation tiers and exports Markdown, LaTeX, JSON, calibration, and plot figures."""
    if severities is None:
        severities = [0, 1, 2, 3]

    if dataset is None:
        dataset = generate_synthetic_benchmark_dataset()

    if evaluator is None:
        evaluator = VLMSpatialEvaluator()

    aggregator = SpatialMetricsAggregator()
    calibration_tracker = CalibrationTracker(n_bins=10)
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

            if is_autoregressive:
                labels = sample.get("labels", {"obj_a": ("blue", "square"), "obj_b": ("red", "square")})
                raw_out = evaluator.evaluate_sample(
                    degraded_img,
                    sample["target_relation"],
                    labels["obj_a"],
                    labels["obj_b"],
                )
                is_correct = bool(raw_out["is_correct"]) if isinstance(raw_out, dict) else bool(raw_out)
                confidence = _extract_prediction_confidence(raw_out)
            else:
                result = evaluator.score_pair(
                    degraded_img, sample["positive_prompt"], sample["negative_prompt"]
                )
                is_correct = bool(result["is_correct"])
                confidence = _extract_prediction_confidence(result)

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
            calibration_tracker.add_prediction(
                severity=severity,
                is_correct=is_correct,
                confidence=confidence,
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
        cal_file = out_path / "calibration_summary.json"

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(aggregator.to_markdown_table())

        aggregator.export_latex(str(latex_file))
        aggregator.export_json(str(json_file))
        calibration_tracker.export_json(str(cal_file))

        reports_generated = {
            "markdown": str(md_file),
            "latex": str(latex_file),
            "json": str(json_file),
            "calibration": str(cal_file),
        }

    if save_plots:
        figures_dir = out_path / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        curve_file = figures_dir / "robustness_curves.png"
        curve_payload = {
            "Model Evaluator": {
                "accuracies": severity_accuracies,
                "color": "#16a34a" if is_autoregressive else "#2563eb",
                "marker": "^" if is_autoregressive else "o",
                "linestyle": "--" if is_autoregressive else "-",
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
        "calibration_tracker": calibration_tracker,
        "clean_accuracy": aggregator.get_clean_accuracy(),
        "perturbed_accuracy": aggregator.get_perturbed_accuracy(),
        "robustness_drop": aggregator.compute_robustness_drop(),
        "reports": reports_generated,
        "plots": plots_generated,
    }


def _build_counterfactual_scorer(
    evaluator: Any,
) -> Callable[[Image.Image, str, str], bool]:
    """Adapt contrastive, autoregressive, or mock evaluators into a binary pair scorer."""

    def _score(image: Image.Image, pos_prompt: str, neg_prompt: str) -> bool:
        # 1. Contrastive dual-encoder path (VLMSpatialEvaluator.score_pair)
        if hasattr(evaluator, "score_pair"):
            result = evaluator.score_pair(image, pos_prompt, neg_prompt)
            if isinstance(result, dict):
                if "is_correct" in result:
                    return bool(result["is_correct"])
                if "margin" in result:
                    return float(result["margin"]) > 0.0
                if "pos_score" in result and "neg_score" in result:
                    return float(result["pos_score"]) > float(result["neg_score"])
                if "correct" in result:
                    return bool(result["correct"])
            if isinstance(result, (tuple, list)) and len(result) >= 2:
                return float(result[0]) > float(result[1])
            if isinstance(result, (int, float)):
                return float(result) > 0.0
            return bool(result)

        # 2. Autoregressive log-prob Yes/No helper path
        if hasattr(evaluator, "score_yes_no"):
            question = f"Is the following statement true? {pos_prompt}"
            res = evaluator.score_yes_no(image, question)
            if isinstance(res, dict):
                return float(res.get("margin", 0.0)) > 0.0
            return float(res) > 0.0

        # 3. AutoregressiveSpatialEvaluator.evaluate_sample path
        if hasattr(evaluator, "evaluate_sample"):
            rel_map = {
                "left of": "left_of",
                "right of": "right_of",
                "in front of": "in_front_of",
                "behind": "behind",
            }
            target_rel = next(
                (code for phrase, code in rel_map.items() if phrase in pos_prompt.lower()),
                "left_of",
            )
            try:
                return bool(
                    evaluator.evaluate_sample(
                        image, target_rel, ("red", "box"), ("blue", "box")
                    )
                )
            except (KeyError, ValueError):
                return False

        # 4. Generic prompt scorer fallback
        if hasattr(evaluator, "score_prompt"):
            return float(evaluator.score_prompt(image, pos_prompt)) > float(
                evaluator.score_prompt(image, neg_prompt)
            )

        raise AttributeError(
            f"Evaluator {type(evaluator).__name__} does not expose a supported scoring method."
        )

    return _score


def run_counterfactual_audit(
    evaluator: Any,
    samples_per_rel: int = 2,
    report_dir: str = "reports",
) -> Dict[str, Any]:
    """Execute horizontal mirror-symmetry and z-order occlusion diagnostics."""
    pairs: List[CounterfactualFlipPair] = []
    for idx in range(max(1, samples_per_rel) * 2):
        img = Image.new("RGB", (224, 224), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        if idx % 2 == 0:
            subj_box, obj_box, rel = (28, 90, 68, 130), (152, 90, 192, 130), "left_of"
        else:
            subj_box, obj_box, rel = (152, 90, 192, 130), (28, 90, 68, 130), "right_of"
        draw.rectangle(subj_box, fill=(220, 50, 47))
        draw.rectangle(obj_box, fill=(38, 139, 210))
        pairs.append(generate_counterfactual_flip_pair(img, subj_box, obj_box, rel))

    scorer_fn = _build_counterfactual_scorer(evaluator)
    symmetry_report = CounterfactualSymmetryAuditor(scorer_fn).evaluate_pairs(pairs)

    renderer = OcclusionSceneRenderer(image_size=224)
    occlusion_samples = [
        renderer.render_pair(subject_in_front=True),
        renderer.render_pair(subject_in_front=False),
    ]
    occ_correct = 0
    for sample in occlusion_samples:
        pos_rel = sample.relation.replace("_", " ")
        neg_rel = "behind" if sample.relation == "in_front_of" else "in front of"
        pos_prompt = f"The {sample.subject_label} is {pos_rel} the {sample.object_label}."
        neg_prompt = f"The {sample.subject_label} is {neg_rel} the {sample.object_label}."
        if scorer_fn(sample.image, pos_prompt, neg_prompt):
            occ_correct += 1

    occlusion_accuracy = occ_correct / len(occlusion_samples)

    summary: Dict[str, Any] = {
        "total_mirror_pairs": symmetry_report.total_pairs,
        "original_accuracy": round(symmetry_report.original_accuracy, 4),
        "counterfactual_accuracy": round(symmetry_report.counterfactual_accuracy, 4),
        "symmetry_consistency": round(symmetry_report.symmetry_consistency, 4),
        "directional_bias": round(symmetry_report.directional_bias, 4),
        "occlusion_accuracy": round(occlusion_accuracy, 4),
    }

    print("\n### Counterfactual Mirror-Symmetry & Depth-Occlusion Audit\n")
    print("| Metric | Value |")
    print("|:---|:---|")
    print(f"| Original Horizontal Accuracy | {summary['original_accuracy'] * 100:.1f}% |")
    print(f"| Counterfactual Flipped Accuracy | {summary['counterfactual_accuracy'] * 100:.1f}% |")
    print(f"| Mirror Symmetry Consistency | {summary['symmetry_consistency'] * 100:.1f}% |")
    print(f"| Left/Right Directional Bias | {summary['directional_bias']:.2f} |")
    print(f"| Depth Occlusion Accuracy (`in_front_of`/`behind`) | {summary['occlusion_accuracy'] * 100:.1f}% |")

    out_path = Path(report_dir) / "counterfactual_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nCounterfactual audit report saved to: {out_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM Spatial Reasoning Benchmark Suite")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory for exported reports")
    parser.add_argument("--severities", nargs="+", type=int, default=[0, 1, 2, 3], help="Photometric degradation levels")
    parser.add_argument("--samples-per-rel", type=int, default=2, help="Number of synthetic samples per relation")
    parser.add_argument("--device", type=str, default=None, help="Compute target device ('cpu' or 'cuda')")
    parser.add_argument(
        "--evaluator-type",
        type=str,
        choices=["clip", "siglip", "autoregressive"],
        default="clip",
        help="Evaluator architecture type",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Pretrained model identifier override")
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Export robustness degradation curves and failure diagnostic overlays into reports/figures/",
    )
    parser.add_argument(
        "--run-counterfactual",
        action="store_true",
        help="Run horizontal mirror-symmetry and depth-occlusion diagnostic audits",
    )
    args = parser.parse_args()

    is_ar = args.evaluator_type == "autoregressive"

    if is_ar:
        model_id = args.model_name or "HuggingFaceTB/SmolVLM-256M-Instruct"
        print(f"Initializing AutoregressiveSpatialEvaluator ({model_id})...")
        evaluator = AutoregressiveSpatialEvaluator(model_name=model_id, device=args.device)
    else:
        model_id = args.model_name or ("openai/clip-vit-base-patch32" if args.evaluator_type == "clip" else "google/siglip-base-patch16-224")
        print(f"Initializing VLMSpatialEvaluator ({model_id})...")
        evaluator = VLMSpatialEvaluator(model_name=model_id, device=args.device)

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
        is_autoregressive=is_ar,
    )

    print("\n" + results["aggregator"].generate_markdown_report())
    print("\n### Confidence Calibration Across Degradation Tiers\n")
    print(results["calibration_tracker"].to_markdown_table())
    print(f"\nBenchmark reports successfully saved to: {args.output_dir}/")
    if args.save_plots:
        print(f"Diagnostic plots saved to: {args.output_dir}/figures/")

    if args.run_counterfactual:
        run_counterfactual_audit(
            evaluator=evaluator,
            samples_per_rel=args.samples_per_rel,
            report_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()