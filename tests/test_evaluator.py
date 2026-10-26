"""Tests for VLMSpatialEvaluator cross-modal alignment and directional topological reasoning."""

from PIL import Image
import pytest

from src.evaluator import VLMSpatialEvaluator
from src.directional_evaluator import DirectionalEvaluator, SpatialRelation, BoundingBox


def test_evaluator_score_pair_structure():
    evaluator = VLMSpatialEvaluator(model_name="openai/clip-vit-base-patch32", device="cpu")
    img = Image.new("RGB", (64, 64), color="red")

    result = evaluator.score_pair(
        image=img,
        positive_text="a red square",
        negative_text="a blue triangle",
    )

    assert "sim_positive" in result
    assert "sim_negative" in result
    assert "is_correct" in result
    assert "margin" in result
    assert isinstance(result["is_correct"], bool)
    assert isinstance(result["margin"], float)
    assert -1.0 <= result["sim_positive"] <= 1.0
    assert -1.0 <= result["sim_negative"] <= 1.0


def test_evaluator_dataset_execution():
    evaluator = VLMSpatialEvaluator(model_name="openai/clip-vit-base-patch32", device="cpu")
    dummy_sample = {
        "image": Image.new("RGB", (64, 64), color="blue"),
        "positive_prompt": "a blue square",
        "negative_prompt": "a red circle",
    }

    metrics = evaluator.evaluate_dataset_at_severity([dummy_sample], severity=0)

    assert metrics["severity_level"] == 0
    assert metrics["total_samples"] == 1
    assert "accuracy" in metrics
    assert "mean_margin" in metrics


def test_evaluator_directional_relation_integration():
    """Verify that VLMSpatialEvaluator evaluates pairwise and batch directional relations."""
    evaluator = VLMSpatialEvaluator.__new__(VLMSpatialEvaluator)
    evaluator.directional_evaluator = DirectionalEvaluator()

    subj = (0.0, 10.0, 10.0, 20.0)
    ref = (20.0, 10.0, 30.0, 20.0)

    # Pairwise evaluation
    res = evaluator.evaluate_spatial_relation(subj, ref, target_relation=SpatialRelation.LEFT_OF)
    assert res["is_satisfied"] is True
    assert "left_of" in res["relations"]
    assert "disjoint" in res["relations"]

    # Batch directional evaluation
    batch_preds = [(subj, ref), ((10.0, 0.0, 20.0, 10.0), (10.0, 20.0, 20.0, 30.0))]
    batch_targets = [SpatialRelation.LEFT_OF, SpatialRelation.ABOVE]
    batch_metrics = evaluator.evaluate_directional_batch(batch_preds, batch_targets)
    assert batch_metrics["accuracy"] == 1.0
    assert batch_metrics["total_evaluated"] == 2.0


def test_evaluator_dataset_execution_with_directional_metadata(monkeypatch):
    """Verify evaluate_dataset_at_severity aggregates directional metrics when sample metadata exists."""
    evaluator = VLMSpatialEvaluator.__new__(VLMSpatialEvaluator)
    evaluator.directional_evaluator = DirectionalEvaluator()

    # Mock score_pair to avoid redundant CLIP model inference during dataset aggregation
    monkeypatch.setattr(
        evaluator,
        "score_pair",
        lambda img, pos, neg: {
            "sim_positive": 0.85,
            "sim_negative": 0.15,
            "is_correct": True,
            "margin": 0.70,
        },
    )

    dummy_img = Image.new("RGB", (64, 64), color="white")
    dataset = [
        {
            "image": dummy_img,
            "positive_prompt": "a green box to the left of a yellow cylinder",
            "negative_prompt": "a green box to the right of a yellow cylinder",
            "subject_box": (0.0, 0.0, 10.0, 10.0),
            "reference_box": (20.0, 0.0, 30.0, 10.0),
            "target_relation": SpatialRelation.LEFT_OF,
        }
    ]

    metrics = evaluator.evaluate_dataset_at_severity(dataset, severity=0)

    assert metrics["severity_level"] == 0
    assert metrics["total_samples"] == 1
    assert metrics["accuracy"] == 100.0
    assert "directional_accuracy" in metrics
    assert metrics["directional_accuracy"] == 100.0
    assert metrics["directional_evaluated"] == 1