"""Integration tests for counterfactual and occlusion audits in run_benchmark.py."""

import json
import numpy as np
from PIL import Image
import pytest

from src.run_benchmark import _build_counterfactual_scorer, run_counterfactual_audit


class _MockGroundedContrastiveEvaluator:
    """Mock contrastive evaluator using red/blue centroid positions and z-layer color."""

    def score_pair(self, image: Image.Image, pos_prompt: str, _neg_prompt: str) -> dict:
        arr = np.asarray(image)
        if "in front of" in pos_prompt or "behind" in pos_prompt:
            red_on_top = arr[112, 112, 0] > 180
            is_true = red_on_top if "in front of" in pos_prompt else not red_on_top
            return {"margin": 0.45 if is_true else -0.45}

        red_x = np.where(arr[:, :, 0] > 180)[1].mean()
        blue_x = np.where(arr[:, :, 2] > 180)[1].mean()
        truth = "left of" if red_x < blue_x else "right of"
        is_true = truth in pos_prompt
        return {"margin": 0.50 if is_true else -0.50}


class _MockAutoregressiveEvaluator:
    """Mock autoregressive evaluator exposing score_yes_no."""

    def score_yes_no(self, _image: Image.Image, question: str) -> dict:
        return {"margin": 0.35 if "left of" in question else -0.35}


def test_run_counterfactual_audit_exports_report(tmp_path):
    evaluator = _MockGroundedContrastiveEvaluator()
    summary = run_counterfactual_audit(
        evaluator=evaluator,
        samples_per_rel=2,
        report_dir=str(tmp_path),
    )

    assert summary["total_mirror_pairs"] == 4
    assert summary["original_accuracy"] == pytest.approx(1.0)
    assert summary["counterfactual_accuracy"] == pytest.approx(1.0)
    assert summary["symmetry_consistency"] == pytest.approx(1.0)
    assert summary["directional_bias"] == pytest.approx(0.0)
    assert summary["occlusion_accuracy"] == pytest.approx(1.0)

    json_file = tmp_path / "counterfactual_audit.json"
    assert json_file.exists()
    loaded = json.loads(json_file.read_text(encoding="utf-8"))
    assert loaded == summary


def test_build_counterfactual_scorer_supports_autoregressive():
    scorer = _build_counterfactual_scorer(_MockAutoregressiveEvaluator())
    dummy_img = Image.new("RGB", (64, 64), (255, 255, 255))
    assert scorer(dummy_img, "The red box is left of the blue box.", "neg") is True
    assert scorer(dummy_img, "The red box is right of the blue box.", "neg") is False