"""Unit tests for diagnostic curves and failure overlay generation."""

from pathlib import Path
from PIL import Image
import pytest

from src.directional_evaluator import BoundingBox
from src.visualizer import plot_robustness_curves, render_spatial_failure_overlay


def test_plot_robustness_curves_export(tmp_path: Path):
    dummy_results = {
        "Model Alpha": {
            "accuracies": [100.0, 85.0, 60.0],
            "color": "#2563eb",
            "marker": "o",
            "linestyle": "-",
        },
        "Model Beta": {
            "accuracies": [90.0, 70.0, 45.0],
            "color": "#dc2626",
            "marker": "s",
            "linestyle": "--",
        },
    }
    severities = [0, 1, 2]
    out_file = tmp_path / "reports" / "figures" / "test_curve.png"

    saved_path = plot_robustness_curves(dummy_results, severities, output_path=str(out_file))

    assert Path(saved_path).exists()
    assert Path(saved_path).stat().st_size > 0


def test_render_spatial_failure_overlay_tuple_boxes(tmp_path: Path):
    base_img = Image.new("RGB", (128, 128), color="white")
    sub_box = (10.0, 10.0, 40.0, 40.0)
    ref_box = (60.0, 60.0, 90.0, 90.0)
    out_file = tmp_path / "diagnostics" / "failure_sample.png"

    overlay = render_spatial_failure_overlay(
        image=base_img,
        subject_box=sub_box,
        reference_box=ref_box,
        target_relation="left_of",
        predicted_relation="right_of",
        output_path=str(out_file),
    )

    assert out_file.exists()
    assert overlay.size == (128, 128)
    assert overlay.mode == "RGB"


def test_render_spatial_failure_overlay_with_bounding_box_dataclass():
    base_img = Image.new("RGB", (64, 64), color="black")
    sub_box = BoundingBox(5.0, 5.0, 20.0, 20.0)
    ref_box = BoundingBox(30.0, 30.0, 50.0, 50.0)

    overlay = render_spatial_failure_overlay(
        image=base_img,
        subject_box=sub_box,
        reference_box=ref_box,
        target_relation="above",
        predicted_relation="above",
    )

    assert overlay.size == (64, 64)