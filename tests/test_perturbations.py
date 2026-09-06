import numpy as np
from PIL import Image
from src.perturbations import PhotometricDegradationEngine


def test_severity_zero_preserves_input():
    dummy = np.full((64, 64, 3), fill_value=128, dtype=np.uint8)
    clean = PhotometricDegradationEngine.apply_low_light(dummy, severity=0)
    assert np.array_equal(dummy, clean), "Severity 0 must preserve pixel values identically."


def test_low_light_severity_monotonic_decrease():
    dummy = np.full((100, 100, 3), fill_value=200, dtype=np.uint8)
    l1 = PhotometricDegradationEngine.apply_low_light(dummy, severity=1)
    l2 = PhotometricDegradationEngine.apply_low_light(dummy, severity=2)
    l3 = PhotometricDegradationEngine.apply_low_light(dummy, severity=3)
    l4 = PhotometricDegradationEngine.apply_low_light(dummy, severity=4)

    assert l1.mean() < dummy.mean(), "Severity 1 must darken the image."
    assert l2.mean() < l1.mean(), "Severity 2 must be darker than Severity 1."
    assert l3.mean() < l2.mean(), "Severity 3 must be darker than Severity 2."
    assert l4.mean() < l3.mean(), "Severity 4 must be darker than Severity 3."
    assert l4.dtype == np.uint8


def test_pipeline_output_type_and_dimensions():
    img = Image.new("RGB", (64, 64), color="white")
    processed = PhotometricDegradationEngine.apply_pipeline(img, severity=2)

    assert isinstance(processed, Image.Image)
    assert processed.size == (64, 64)