from typing import Any, Dict, List
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from src.perturbations import PhotometricDegradationEngine


class VLMSpatialEvaluator:
    """Evaluates zero-shot cross-modal spatial alignment across degradation tiers."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def score_pair(self, image: Image.Image, positive_text: str, negative_text: str) -> Dict[str, Any]:
        """Calculates cosine similarity between visual embedding and paired text descriptions."""
        inputs = self.processor(
            text=[positive_text, negative_text],
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Forward pass yields projected multimodal embeddings
        outputs = self.model(**inputs)

        # Extract projected tensor embeddings
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
        else:
            # Fallback extraction if model outputs container objects
            raw_img = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            raw_txt = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
            )
            image_features = raw_img.pooler_output if hasattr(raw_img, "pooler_output") and raw_img.pooler_output is not None else raw_img
            text_features = raw_txt.pooler_output if hasattr(raw_txt, "pooler_output") and raw_txt.pooler_output is not None else raw_txt

        # L2-normalize visual and textual latent vectors
        image_norm = F.normalize(image_features, p=2, dim=-1)
        text_norm = F.normalize(text_features, p=2, dim=-1)

        # Cosine similarity: (1, dim) @ (dim, 2) -> (1, 2)
        similarities = torch.matmul(image_norm, text_norm.T).squeeze(0).cpu().tolist()
        sim_pos, sim_neg = float(similarities[0]), float(similarities[1])

        return {
            "sim_positive": sim_pos,
            "sim_negative": sim_neg,
            "is_correct": bool(sim_pos > sim_neg),
            "margin": float(sim_pos - sim_neg),
        }

    def evaluate_dataset_at_severity(
        self, dataset: List[Dict[str, Any]], severity: int
    ) -> Dict[str, Any]:
        """Evaluates spatial accuracy over an entire dataset split at a given degradation level."""
        correct = 0
        margins = []

        for sample in dataset:
            degraded_img = PhotometricDegradationEngine.apply_pipeline(
                sample["image"], severity=severity
            )
            result = self.score_pair(
                degraded_img, sample["positive_prompt"], sample["negative_prompt"]
            )
            if result["is_correct"]:
                correct += 1
            margins.append(result["margin"])

        accuracy = (correct / len(dataset)) * 100.0 if dataset else 0.0
        avg_margin = sum(margins) / len(margins) if margins else 0.0

        return {
            "severity_level": severity,
            "total_samples": len(dataset),
            "accuracy": round(accuracy, 2),
            "mean_margin": round(avg_margin, 4),
        }