"""Integration tests for automated benchmark pipeline runner, reports, and diagnostic plots."""

from pathlib import Path
import pytest

from src.directional_evaluator import DirectionalEvaluator
from src.evaluator import VLMSpatialEvaluator
from src.run_benchmark import generate_synthetic_benchmark_dataset, run_benchmark_pipeline


def test_generate_synthetic_benchmark_dataset():
    dataset = generate_synthetic_benchmark_dataset(samples_per_relation=2)
    assert len(dataset) == 10

    for sample in dataset:
        assert "id" in sample
        assert "image" in sample
        assert "positive_prompt" in sample
        assert "negative_prompt" in sample
        assert "subject_box" in sample
        assert "reference_box" in sample
        assert "target_relation" in sample


def test_run_benchmark_pipeline_mocked(tmp_path: Path):
    evaluator = VLMSpatialEvaluator.__new__(VLMSpatialEvaluator)
    evaluator.directional_evaluator = DirectionalEvaluator()

    evaluator.score_pair = lambda img, pos, neg: {
        "sim_positive": 0.88,
        "sim_negative": 0.12,
        "is_correct": True,
        "margin": 0.76,
    }

    out_dir = tmp_path / "benchmark_run"
    dataset = generate_synthetic_benchmark_dataset(samples_per_relation=1)

    results = run_benchmark_pipeline(
        evaluator=evaluator,
        dataset=dataset,
        output_dir=str(out_dir),
        severities=[0, 1],
        save_reports=True,
        save_plots=False,
    )

    assert results["clean_accuracy"] == 1.0
    assert results["perturbed_accuracy"] == 1.0
    assert results["robustness_drop"] == 0.0

    md_path = out_dir / "benchmark_summary.md"
    tex_path = out_dir / "benchmark_report.tex"
    json_path = out_dir / "benchmark_summary.json"

    assert md_path.exists()
    assert tex_path.exists()
    assert json_path.exists()

    md_content = md_path.read_text(encoding="utf-8")
    assert "Spatial Reasoning & Robustness Benchmark Summary" in md_content
    assert "| `left_of` |" in md_content


def test_run_benchmark_pipeline_with_plots(tmp_path: Path):
    evaluator = VLMSpatialEvaluator.__new__(VLMSpatialEvaluator)
    evaluator.directional_evaluator = DirectionalEvaluator()

    # Alternate correct and incorrect to trigger failure overlays
    toggle = {"flag": True}

    def alternating_score(img, pos, neg):
        toggle["flag"] = not toggle["flag"]
        return {
            "sim_positive": 0.7 if toggle["flag"] else 0.3,
            "sim_negative": 0.3 if toggle["flag"] else 0.7,
            "is_correct": toggle["flag"],
            "margin": 0.4 if toggle["flag"] else -0.4,
        }

    evaluator.score_pair = alternating_score

    out_dir = tmp_path / "benchmark_plots_run"
    dataset = generate_synthetic_benchmark_dataset(samples_per_relation=1)

    results = run_benchmark_pipeline(
        evaluator=evaluator,
        dataset=dataset,
        output_dir=str(out_dir),
        severities=[0, 1],
        save_reports=True,
        save_plots=True,
    )

    curve_path = out_dir / "figures" / "robustness_curves.png"
    failures_dir = out_dir / "figures" / "failures"

    assert curve_path.exists()
    assert curve_path.stat().st_size > 0
    assert failures_dir.exists()
    assert len(list(failures_dir.glob("*.png"))) > 0
    assert "curve" in results["plots"]
    assert len(results["plots"]["failures"]) > 0