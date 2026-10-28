"""Integration tests for automated benchmark pipeline runner and artifact generation."""

from pathlib import Path
import pytest

from src.directional_evaluator import DirectionalEvaluator
from src.evaluator import VLMSpatialEvaluator
from src.run_benchmark import generate_synthetic_benchmark_dataset, run_benchmark_pipeline


def test_generate_synthetic_benchmark_dataset():
    dataset = generate_synthetic_benchmark_dataset(samples_per_relation=2)
    # 5 relations * 2 samples = 10 samples
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
    # Instantiate evaluator without downloading model weights for unit tests
    evaluator = VLMSpatialEvaluator.__new__(VLMSpatialEvaluator)
    evaluator.directional_evaluator = DirectionalEvaluator()

    # Deterministic inference mock
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
    )

    assert results["clean_accuracy"] == 1.0
    assert results["perturbed_accuracy"] == 1.0
    assert results["robustness_drop"] == 0.0

    # Verify report files were generated on disk
    md_path = out_dir / "benchmark_summary.md"
    tex_path = out_dir / "benchmark_report.tex"
    json_path = out_dir / "benchmark_summary.json"

    assert md_path.exists()
    assert tex_path.exists()
    assert json_path.exists()

    md_content = md_path.read_text(encoding="utf-8")
    assert "Spatial Reasoning & Robustness Benchmark Summary" in md_content
    assert "| `left_of` |" in md_content

    tex_content = tex_path.read_text(encoding="utf-8")
    assert r"\begin{table}" in tex_content
    assert r"\toprule" in tex_content