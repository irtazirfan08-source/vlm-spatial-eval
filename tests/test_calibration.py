"""Unit tests for VLM confidence calibration metrics (ECE, MCE, Overconfidence, Brier)."""

import json
import pytest

from src.calibration import (
    CalibrationTracker,
    compute_brier_score,
    compute_expected_calibration_error,
    margin_to_confidence,
    margin_to_probability,
)


def test_margin_to_probability_and_confidence():
    assert margin_to_probability(0.0) == pytest.approx(0.5)
    assert margin_to_confidence(0.0) == pytest.approx(0.5)

    pos_conf = margin_to_confidence(2.0)
    neg_conf = margin_to_confidence(-2.0)
    assert pos_conf == pytest.approx(neg_conf)
    assert 0.85 < pos_conf < 0.90

    # Extreme logits should clamp cleanly without OverflowError
    assert margin_to_probability(1000.0) == pytest.approx(1.0)
    assert margin_to_probability(-1000.0) == pytest.approx(0.0)


def test_perfect_calibration_yields_zero_ece():
    # Bin [0.2, 0.3): 10 predictions at 0.25 with 25% empirical accuracy (not binary, or exact match)
    # Using exact bin match: 5 predictions at 0.80 with 4 correct (80% accuracy)
    confs = [0.80] * 5
    corrs = [1, 1, 1, 1, 0]

    summary = compute_expected_calibration_error(confs, corrs, n_bins=10)
    assert summary.accuracy == pytest.approx(0.80)
    assert summary.mean_confidence == pytest.approx(0.80)
    assert summary.ece == pytest.approx(0.0)
    assert summary.mce == pytest.approx(0.0)
    assert summary.overconfidence_error == pytest.approx(0.0)


def test_overconfident_collapse_high_ece_and_brier():
    confs = [0.90] * 10
    corrs = [False] * 10

    summary = compute_expected_calibration_error(confs, corrs, n_bins=10)
    assert summary.accuracy == pytest.approx(0.0)
    assert summary.ece == pytest.approx(0.90)
    assert summary.mce == pytest.approx(0.90)
    assert summary.overconfidence_error == pytest.approx(0.90)
    assert summary.brier_score == pytest.approx(0.81)


def test_brier_score_mixed_predictions():
    confs = [1.0, 0.0, 0.8, 0.2]
    corrs = [True, False, True, False]
    # Squared errors: 0.0, 0.0, 0.04, 0.04 -> mean = 0.02
    assert compute_brier_score(confs, corrs) == pytest.approx(0.02)


def test_calibration_tracker_reports_and_json_export(tmp_path):
    tracker = CalibrationTracker(n_bins=5)
    tracker.add_prediction(severity=0, is_correct=True, margin=2.2)
    tracker.add_prediction(severity=0, is_correct=True, margin=1.8)
    tracker.add_prediction(severity=1, is_correct=False, confidence=0.85)
    tracker.add_prediction(severity=1, is_correct=True, confidence=0.75)

    md = tracker.to_markdown_table()
    assert "| L0 |" in md
    assert "| L1 |" in md
    assert "ECE" in md

    latex = tracker.to_latex_table()
    assert r"\begin{tabular}" in latex
    assert r"\bottomrule" in latex

    out_json = tmp_path / "calibration_summary.json"
    tracker.export_json(str(out_json))
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert set(loaded.keys()) == {"L0", "L1"}
    assert loaded["L0"]["accuracy"] == pytest.approx(1.0)
    assert loaded["L1"]["overconfidence_error"] > 0.0


def test_calibration_input_validation():
    with pytest.raises(ValueError):
        compute_expected_calibration_error([], [])
    with pytest.raises(ValueError):
        compute_expected_calibration_error([0.5, 0.6], [1])
    with pytest.raises(ValueError):
        compute_expected_calibration_error([1.2], [1])
    with pytest.raises(ValueError):
        margin_to_probability(1.0, temperature=0.0)