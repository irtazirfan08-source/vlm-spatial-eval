import numpy as np
from PIL import Image
from src.linear_probe import ViTSpatialProbe


def test_probe_extraction_shape():
    probe = ViTSpatialProbe(model_name="openai/clip-vit-base-patch32", device="cpu")
    dummy_images = [
        Image.new("RGB", (64, 64), color="red"),
        Image.new("RGB", (64, 64), color="blue"),
    ]

    embeddings = probe.extract_cls_embeddings(dummy_images, layer_index=-1)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2  # Batch size
    assert embeddings.shape[1] == 768  # ViT-B/32 hidden dimension


def test_probe_train_and_evaluate():
    # Synthetic linearly separable features
    x_train = np.array([[2.0, 3.0], [3.0, 4.0], [-2.0, -3.0], [-3.0, -4.0]])
    y_train = [1, 1, 0, 0]

    x_test = np.array([[2.5, 3.5], [-2.5, -3.5]])
    y_test = [1, 0]

    accuracy = ViTSpatialProbe.train_and_evaluate(x_train, y_train, x_test, y_test)

    assert accuracy == 100.0