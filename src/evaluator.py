"""VLM Spatial Evaluator integrating CLIP similarity, degradation pipelines, and directional topological reasoning."""

from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

from .perturbations import PhotometricDegradationEngine
from .directional_evaluator import BoundingBox, DirectionalEvaluator, SpatialRelation


class VLMSpatialEvaluator:
    """Evaluates zero-shot cross-modal spatial alignment across degradation tiers."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.directional_evaluator = DirectionalEvaluator()

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

    def evaluate_spatial_relation(
        self,
        subject: Union[BoundingBox, Tuple[float, float, float, float]],
        reference: Union[BoundingBox, Tuple[float, float, float, float]],
        target_relation: Optional[Union[SpatialRelation, str]] = None,
        tolerance: float = 0.0,
    ) -> Dict[str, Any]:
        """Evaluates topological and directional relations between two bounding boxes."""
        subj_box = subject if isinstance(subject, BoundingBox) else BoundingBox(*subject)
        ref_box = reference if isinstance(reference, BoundingBox) else BoundingBox(*reference)

        detected = self.directional_evaluator.evaluate_pairwise_relation(
            subj_box, ref_box, tolerance=tolerance
        )
        iou = self.directional_evaluator.compute_iou(subj_box, ref_box)

        res: Dict[str, Any] = {
            "iou": iou,
            "relations": [r.value if isinstance(r, SpatialRelation) else r for r in detected],
            "detected_relations": detected,
        }

        if target_relation is not None:
            target_enum = (
                target_relation
                if isinstance(target_relation, SpatialRelation)
                else SpatialRelation(target_relation)
            )
            res["target_relation"] = target_enum.value
            res["is_satisfied"] = target_enum in detected

        return res

    def evaluate_directional_batch(
        self,
        predictions: List[Tuple[Any, Any]],
        ground_truth_relations: List[Union[SpatialRelation, str]],
        tolerance: float = 0.0,
    ) -> Dict[str, Any]:
        """Evaluates batch directional predictions against expected spatial relations."""
        parsed_pairs = []
        for s, r in predictions:
            sb = s if isinstance(s, BoundingBox) else BoundingBox(*s)
            rb = r if isinstance(r, BoundingBox) else BoundingBox(*r)
            parsed_pairs.append((sb, rb))

        parsed_targets = [
            t if isinstance(t, SpatialRelation) else SpatialRelation(t)
            for t in ground_truth_relations
        ]

        return self.directional_evaluator.evaluate_prediction_accuracy(
            parsed_pairs, parsed_targets
        )

    def evaluate_dataset_at_severity(
        self, dataset: List[Dict[str, Any]], severity: int
    ) -> Dict[str, Any]:
        """Evaluates spatial accuracy over an entire dataset split at a given degradation level."""
        correct = 0
        margins = []
        dir_correct = 0
        dir_total = 0

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

            # Evaluate directional relation metadata if present in sample
            if "subject_box" in sample and "reference_box" in sample and "target_relation" in sample:
                rel_eval = self.evaluate_spatial_relation(
                    sample["subject_box"],
                    sample["reference_box"],
                    sample["target_relation"],
                )
                if rel_eval.get("is_satisfied", False):
                    dir_correct += 1
                dir_total += 1

        accuracy = (correct / len(dataset)) * 100.0 if dataset else 0.0
        avg_margin = sum(margins) / len(margins) if margins else 0.0

        output: Dict[str, Any] = {
            "severity_level": severity,
            "total_samples": len(dataset),
            "accuracy": round(accuracy, 2),
            "mean_margin": round(avg_margin, 4),
        }

        if dir_total > 0:
            output["directional_accuracy"] = round((dir_correct / dir_total) * 100.0, 2)
            output["directional_evaluated"] = dir_total

        return output