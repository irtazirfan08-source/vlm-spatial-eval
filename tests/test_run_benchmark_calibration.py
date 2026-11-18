"""Integration tests for CalibrationTracker wiring in run_benchmark.py."""

import json
from pathlib import Path
import pytest

from src.run_benchmark import (
    _extract_prediction_confidence,
    generate_synthetic_benchmark_dataset,
    run_benchmark_pipeline,
)


class _MockOverconfidentContrastiveEvaluator:
    """Simulates a VLM that stays highly confident (~88%) even when degraded predictions fail."""

    def __init__(self):
        self.call_count = 0

    def score_pair(self, _image, _pos_prompt: str, _neg_prompt: str) -> dict:
        self.call_count += 1
        # First 5 calls (severity 0) are right; next 5 calls (severity 1) are wrong
        is_correct = self.call_count <= 5
        margin = 2.0 if is_correct else -2.0
        return {
            "is_correct": is_correct,
            "margin": margin,
        }


def test_extract_prediction_confidence_variants():
    assert _extract_prediction_confidence({"confidence": 0.91}) == pytest.approx(0.91)
    assert 0.85 < _extract_prediction_confidence({"margin": 2.0}) < 0.90
    assert 0.85 < _extract_prediction_confidence({"pos_score": 0.8, "neg_score": -1.2}) < 0.90
    # Fallback when mock omits margin/confidence keys
    assert _extract_prediction_confidence({"is_correct": True}) == pytest.approx(0.75)
    assert _extract_prediction_confidence(False) == pytest.approx(0.75)


def test_run_benchmark_pipeline_exports_calibration_summary(tmp_path):
    dataset = generate_synthetic_benchmark_dataset(samples_per_relation=1)
    evaluator = _MockOverconfidentContrastiveEvaluator()

    results = run_benchmark_pipeline(
        evaluator=evaluator,
        dataset=dataset,
        output_dir=str(tmp_path),
        severities=[0, 1],
        save_reports=True,
        save_plots=False,
    )

    assert "calibration_tracker" in results
    assert "calibration" in results["reports"]

    cal_path = Path(results["reports"]["calibration"])
    assert cal_path.exists()

    payload = json.loads(cal_path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"L0", "L1"}
    assert payload["L0"]["accuracy"] == pytest.approx(1.0)
    assert payload["L1"]["accuracy"] == pytest.approx(0.0)
    assert payload["L1"]["overconfidence_error"] > 0.80