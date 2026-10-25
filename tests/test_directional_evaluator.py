"""Unit tests for directional spatial relation evaluator."""

import pytest
from src.directional_evaluator import BoundingBox, DirectionalEvaluator, SpatialRelation


@pytest.fixture
def evaluator():
    return DirectionalEvaluator()


def test_bounding_box_properties():
    box = BoundingBox(xmin=10.0, ymin=20.0, xmax=30.0, ymax=60.0)
    assert box.width == 20.0
    assert box.height == 40.0
    assert box.area == 800.0
    assert box.center == (20.0, 40.0)


def test_bounding_box_invalid_coordinates():
    with pytest.raises(ValueError):
        BoundingBox(xmin=50.0, ymin=10.0, xmax=20.0, ymax=30.0)


def test_compute_iou(evaluator):
    box_a = BoundingBox(0.0, 0.0, 10.0, 10.0)
    box_b = BoundingBox(0.0, 0.0, 10.0, 10.0)
    assert evaluator.compute_iou(box_a, box_b) == 1.0

    box_c = BoundingBox(20.0, 20.0, 30.0, 30.0)
    assert evaluator.compute_iou(box_a, box_c) == 0.0

    # Overlapping 5x10 area
    box_d = BoundingBox(5.0, 0.0, 15.0, 10.0)
    # intersection: 5*10 = 50, union: 100 + 100 - 50 = 150 -> 50/150 = 1/3
    assert pytest.approx(evaluator.compute_iou(box_a, box_d), abs=1e-5) == (1.0 / 3.0)


def test_directional_relations_left_and_right(evaluator):
    left_box = BoundingBox(0.0, 10.0, 10.0, 20.0)
    right_box = BoundingBox(20.0, 10.0, 30.0, 20.0)

    left_rels = evaluator.evaluate_pairwise_relation(left_box, right_box)
    assert SpatialRelation.LEFT_OF in left_rels
    assert SpatialRelation.DISJOINT in left_rels

    right_rels = evaluator.evaluate_pairwise_relation(right_box, left_box)
    assert SpatialRelation.RIGHT_OF in right_rels
    assert SpatialRelation.DISJOINT in right_rels


def test_directional_relations_above_and_below(evaluator):
    top_box = BoundingBox(10.0, 0.0, 20.0, 10.0)
    bottom_box = BoundingBox(10.0, 20.0, 20.0, 30.0)

    top_rels = evaluator.evaluate_pairwise_relation(top_box, bottom_box)
    assert SpatialRelation.ABOVE in top_rels

    bottom_rels = evaluator.evaluate_pairwise_relation(bottom_box, top_box)
    assert SpatialRelation.BELOW in bottom_rels


def test_containment(evaluator):
    outer = BoundingBox(0.0, 0.0, 50.0, 50.0)
    inner = BoundingBox(10.0, 10.0, 20.0, 20.0)

    assert evaluator.evaluate_containment(inner, outer) is True
    assert evaluator.evaluate_containment(outer, inner) is False

    relations = evaluator.evaluate_pairwise_relation(inner, outer)
    assert SpatialRelation.INSIDE in relations
    assert SpatialRelation.OVERLAPPING in relations


def test_prediction_accuracy_pipeline(evaluator):
    pred_pairs = [
        (BoundingBox(0.0, 0.0, 10.0, 10.0), BoundingBox(20.0, 0.0, 30.0, 10.0)),
        (BoundingBox(10.0, 0.0, 20.0, 10.0), BoundingBox(10.0, 20.0, 20.0, 30.0)),
    ]
    ground_truth = [SpatialRelation.LEFT_OF, SpatialRelation.ABOVE]

    metrics = evaluator.evaluate_prediction_accuracy(pred_pairs, ground_truth)
    assert metrics["accuracy"] == 1.0
    assert metrics["total_evaluated"] == 2.0