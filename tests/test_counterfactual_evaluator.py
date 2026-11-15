"""Unit tests for counterfactual mirror symmetry and depth-occlusion predicates."""

from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw
import pytest

from src.counterfactual_evaluator import (
    CounterfactualSymmetryAuditor,
    OcclusionSceneRenderer,
    compute_bbox_iou,
    compute_occlusion_ratio,
    evaluate_occlusion_relation,
    flip_bbox_horizontally,
    generate_counterfactual_flip_pair,
    invert_horizontal_relation,
)


@dataclass
class _DummyBox:
    x1: float
    y1: float
    x2: float
    y2: float


def test_flip_bbox_horizontally_preserves_dimensions():
    flipped = flip_bbox_horizontally((20, 40, 60, 100), image_width=224)
    assert flipped == (164.0, 40.0, 204.0, 100.0)
    assert (flipped[2] - flipped[0]) == 40.0
    assert (flipped[3] - flipped[1]) == 60.0

    # Verify duck-typed dataclass support
    dummy = _DummyBox(10, 20, 50, 80)
    assert flip_bbox_horizontally(dummy, image_width=200) == (150.0, 20.0, 190.0, 80.0)


def test_invert_horizontal_relation_and_invariants():
    assert invert_horizontal_relation("left_of") == "right_of"
    assert invert_horizontal_relation("right_of") == "left_of"
    for invariant in ("above", "below", "inside", "in_front_of", "behind"):
        assert invert_horizontal_relation(invariant) == invariant


def test_counterfactual_flip_pair_pixel_alignment():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    subj_box = (20, 80, 60, 120)
    obj_box = (140, 80, 180, 120)
    draw.rectangle(subj_box, fill=(255, 0, 0))
    draw.rectangle(obj_box, fill=(0, 0, 255))

    pair = generate_counterfactual_flip_pair(img, subj_box, obj_box, "left_of")
    assert pair.flipped_relation == "right_of"
    assert pair.flipped_subject_bbox == (140.0, 80.0, 180.0, 120.0)
    assert pair.flipped_object_bbox == (20.0, 80.0, 60.0, 120.0)

    # Confirm mirrored image pixels match the flipped bounding boxes
    arr = np.asarray(pair.flipped_image)
    assert tuple(arr[100, 160]) == (255, 0, 0)
    assert tuple(arr[100, 40]) == (0, 0, 255)


def test_symmetry_auditor_distinguishes_grounded_vs_biased():
    pairs = []
    for rel, s_box, o_box in [
        ("left_of", (20, 80, 60, 120), (140, 80, 180, 120)),
        ("right_of", (140, 80, 180, 120), (20, 80, 60, 120)),
    ]:
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle(s_box, fill=(255, 0, 0))
        draw.rectangle(o_box, fill=(0, 0, 255))
        pairs.append(generate_counterfactual_flip_pair(img, s_box, o_box, rel))

    def grounded_scorer(image: Image.Image, pos_prompt: str, _neg: str) -> bool:
        arr = np.asarray(image)
        red_x = np.where(arr[:, :, 0] > 200)[1].mean()
        blue_x = np.where(arr[:, :, 2] > 200)[1].mean()
        truth = "left of" if red_x < blue_x else "right of"
        return truth in pos_prompt

    def lazy_biased_scorer(_image: Image.Image, pos_prompt: str, _neg: str) -> bool:
        return "left of" in pos_prompt

    grounded_rep = CounterfactualSymmetryAuditor(grounded_scorer).evaluate_pairs(pairs)
    assert grounded_rep.original_accuracy == pytest.approx(1.0)
    assert grounded_rep.counterfactual_accuracy == pytest.approx(1.0)
    assert grounded_rep.symmetry_consistency == pytest.approx(1.0)
    assert grounded_rep.directional_bias == pytest.approx(0.0)

    biased_rep = CounterfactualSymmetryAuditor(lazy_biased_scorer).evaluate_pairs(pairs)
    assert biased_rep.original_accuracy == pytest.approx(0.5)
    assert biased_rep.counterfactual_accuracy == pytest.approx(0.5)
    assert biased_rep.symmetry_consistency == pytest.approx(0.0)
    assert biased_rep.directional_bias == pytest.approx(1.0)


def test_occlusion_predicates_and_z_order_rendering():
    box_a = (40, 40, 120, 120)
    box_b = (80, 80, 160, 160)
    far_box = (170, 170, 210, 210)

    assert compute_bbox_iou(box_a, box_b) > 0.1
    assert compute_occlusion_ratio(box_a, box_b) == pytest.approx(0.25)
    assert evaluate_occlusion_relation(box_a, subject_z=2, object_bbox=box_b, object_z=1) == "in_front_of"
    assert evaluate_occlusion_relation(box_a, subject_z=1, object_bbox=box_b, object_z=2) == "behind"
    assert evaluate_occlusion_relation(box_a, subject_z=2, object_bbox=far_box, object_z=1) == "disjoint"

    renderer = OcclusionSceneRenderer(image_size=200)
    front_sample = renderer.render_pair(subject_in_front=True)
    behind_sample = renderer.render_pair(subject_in_front=False)

    assert front_sample.relation == "in_front_of"
    assert behind_sample.relation == "behind"

    # Center overlap pixel (100, 100) must reflect foreground z-layer color
    front_arr = np.asarray(front_sample.image)
    behind_arr = np.asarray(behind_sample.image)
    assert front_arr[100, 100, 0] > 200  # red subject on top
    assert behind_arr[100, 100, 2] > 180  # blue object on top