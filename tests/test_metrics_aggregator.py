import json
import pytest
from pathlib import Path
from src.metrics_aggregator import SpatialMetricsAggregator


def test_metrics_aggregator_basic_accuracy():
    aggregator = SpatialMetricsAggregator()

    aggregator.add_evaluation("s1", "left_of", predicted="left", ground_truth="left", is_perturbed=False)
    aggregator.add_evaluation("s2", "left_of", predicted="left", ground_truth="left", is_perturbed=False)
    aggregator.add_evaluation("s3", "right_of", predicted="right", ground_truth="right", is_perturbed=False)
    aggregator.add_evaluation("s4", "right_of", predicted="left", ground_truth="right", is_perturbed=False)

    assert aggregator.get_clean_accuracy() == pytest.approx(0.75, 0.01)


def test_metrics_aggregator_robustness_degradation():
    aggregator = SpatialMetricsAggregator()

    # Clean samples: 100% correct
    aggregator.add_evaluation("s1", "above", "above", "above", is_perturbed=False)
    aggregator.add_evaluation("s2", "above", "above", "above", is_perturbed=False)

    # Perturbed samples: 50% correct under rotation
    aggregator.add_evaluation("s1_p", "above", "above", "above", is_perturbed=True, perturbation_type="rotation_90")
    aggregator.add_evaluation("s2_p", "above", "below", "above", is_perturbed=True, perturbation_type="rotation_90")

    clean_acc = aggregator.get_clean_accuracy()
    pert_acc = aggregator.get_perturbed_accuracy()
    drop = aggregator.compute_robustness_drop()

    assert clean_acc == 1.0
    assert pert_acc == 0.5
    assert drop == pytest.approx(0.5, 0.01)


def test_per_relation_breakdown_and_markdown():
    aggregator = SpatialMetricsAggregator()
    aggregator.add_evaluation("s1", "behind", "behind", "behind", is_perturbed=False)
    aggregator.add_evaluation("s2", "in_front_of", "in_front_of", "in_front_of", is_perturbed=False)
    aggregator.add_evaluation("s3", "behind", "in_front_of", "behind", is_perturbed=True, perturbation_type="blur")

    breakdown = aggregator.compute_per_relation_breakdown()
    assert "behind" in breakdown
    assert "in_front_of" in breakdown
    assert breakdown["behind"]["clean_accuracy"] == 1.0
    assert breakdown["behind"]["perturbed_accuracy"] == 0.0

    md = aggregator.generate_markdown_report()
    assert "Spatial Reasoning & Robustness Benchmark Summary" in md
    assert "`behind`" in md
    assert "`in_front_of`" in md


def test_export_json(tmp_path: Path):
    aggregator = SpatialMetricsAggregator()
    aggregator.add_evaluation("s1", "left_of", "left", "left", is_perturbed=False)
    aggregator.add_evaluation("s2", "left_of", "right", "left", is_perturbed=True, perturbation_type="crop")

    out_file = tmp_path / "benchmark_summary.json"
    aggregator.export_json(str(out_file))

    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "summary" in data
    assert "per_relation" in data
    assert data["summary"]["total_samples"] == 2
    assert data["summary"]["clean_accuracy"] == 1.0
    assert data["summary"]["perturbed_accuracy"] == 0.0