from typing import Any, List, Tuple
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch
from transformers import AutoProcessor, CLIPVisionModelWithProjection


class ViTSpatialProbe:
    """Extracts intermediate layer activations to evaluate linear separability of spatial features."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.vision_model = (
            CLIPVisionModelWithProjection.from_pretrained(model_name, output_hidden_states=True)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def extract_cls_embeddings(
        self, images: List[Image.Image], layer_index: int = -1
    ) -> np.ndarray:
        """
        Extracts pooled [CLS] token representations from a specific transformer encoder layer.
        layer_index: -1 corresponds to the final encoder layer before projection.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.vision_model(**inputs)

        # hidden_states is a tuple: (embedding_layer, layer_1, ..., layer_N)
        hidden_states = outputs.hidden_states
        target_layer = hidden_states[layer_index]

        # Extract [CLS] token (index 0 across sequence dimension)
        cls_tokens = target_layer[:, 0, :].detach().cpu().numpy()
        return cls_tokens

    @staticmethod
    def train_and_evaluate(
        x_train: np.ndarray,
        y_train: List[int],
        x_test: np.ndarray,
        y_test: List[int],
    ) -> float:
        """Fits a logistic regression probe and returns test accuracy percentage."""
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        return float(accuracy_score(y_test, predictions) * 100.0)