import cv2
import numpy as np
from PIL import Image


class PhotometricDegradationEngine:
    """Applies controlled physical photon starvation and sensor noise."""

    @staticmethod
    def apply_low_light(image: np.ndarray, severity: int) -> np.ndarray:
        """
        Simulates photon starvation via non-linear gamma expansion and intensity scaling.
        Severity 0: Clean image (identity)
        Severity 1-4: Progressive attenuation down to deep underexposure (< 5 lux equivalent)
        """
        if severity == 0:
            return image.copy()

        gamma_table = {1: 1.5, 2: 2.2, 3: 3.0, 4: 4.2}
        scale_table = {1: 0.8, 2: 0.5, 3: 0.25, 4: 0.08}

        gamma = gamma_table.get(severity, 1.0)
        scale = scale_table.get(severity, 1.0)

        # I_out = ((I_in / 255) ^ gamma) * scale * 255
        normalized = image.astype(np.float32) / 255.0
        degraded = np.power(normalized, gamma) * scale
        degraded = np.clip(degraded * 255.0, 0, 255).astype(np.uint8)
        return degraded

    @staticmethod
    def apply_sensor_noise(image: np.ndarray, severity: int) -> np.ndarray:
        """Adds signal-dependent Gaussian sensor read noise."""
        if severity == 0:
            return image.copy()

        sigma_table = {1: 5.0, 2: 15.0, 3: 30.0, 4: 60.0}
        sigma = sigma_table.get(severity, 0.0)

        gaussian_noise = np.random.normal(0, sigma, image.shape)
        noisy = image.astype(np.float32) + gaussian_noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    @classmethod
    def apply_pipeline(cls, pil_image: Image.Image, severity: int) -> Image.Image:
        """Chains degradation operators and returns a PIL Image."""
        arr = np.array(pil_image)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[-1] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)

        arr = cls.apply_low_light(arr, severity)
        arr = cls.apply_sensor_noise(arr, severity)
        return Image.fromarray(arr)