"""Evaluates generative autoregressive VLMs on directional spatial discrimination via token log-probabilities."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import torch
from transformers import AutoProcessor

# Modern transformers uses AutoModelForImageTextToText; fallback ensures cross-version compatibility
try:
    from transformers import AutoModelForImageTextToText as AutoModelForVLM
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as AutoModelForVLM
    except ImportError:
        from transformers import AutoModel as AutoModelForVLM

from .perturbations import PhotometricDegradationEngine


class AutoregressiveSpatialEvaluator:
    """Evaluates generative autoregressive VLMs on directional spatial discrimination using token logprobs or generation."""

    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.bfloat16,
        ).to(self.device).eval()

        # Cache token IDs for "Yes" and "No" for logprob extraction
        # We check both capitalized and lowercase variants across tokenizer vocabularies
        self.yes_token_ids = self._get_token_ids(["Yes", "yes", " YES"])
        self.no_token_ids = self._get_token_ids(["No", "no", " NO"])

    def _get_token_ids(self, candidates: List[str]) -> List[int]:
        ids = []
        for cand in candidates:
            encoded = self.processor.tokenizer.encode(cand, add_special_tokens=False)
            if encoded:
                ids.append(encoded[-1])  # take the last token representation
        return list(set(ids))

    def score_sample_logprobs(
        self, image: Image.Image, relation: str, obj_a: Tuple[str, str], obj_b: Tuple[str, str]
    ) -> Dict[str, Any]:
        """
        Computes conditional log-probabilities for 'Yes' vs 'No' tokens at the first generation step.
        """
        color_a, shape_a = obj_a
        color_b, shape_b = obj_b

        prompt = (
            f"Question: Is the {color_a} {shape_a} {relation} the {color_b} {shape_b}? "
            "Answer with only 'Yes' or 'No'."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt_text, images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Extract logits for the very next predicted token (last position)
            next_token_logits = outputs.logits[0, -1, :]
            log_probs = torch.log_softmax(next_token_logits, dim=-1)

        # Aggregate log probabilities for Yes and No candidates
        yes_logprob = max([log_probs[tid].item() for tid in self.yes_token_ids]) if self.yes_token_ids else -100.0
        no_logprob = max([log_probs[tid].item() for tid in self.no_token_ids]) if self.no_token_ids else -100.0

        is_correct = yes_logprob > no_logprob
        margin = yes_logprob - no_logprob

        return {
            "is_correct": is_correct,
            "yes_logprob": yes_logprob,
            "no_logprob": no_logprob,
            "margin": margin,
        }

    def evaluate_sample(self, image: Image.Image, relation: str, obj_a: tuple, obj_b: tuple) -> bool:
        """
        Evaluates spatial relation using token logprob scoring with a fallback to generative decoding.
        """
        try:
            res = self.score_sample_logprobs(image, relation, obj_a, obj_b)
            return res["is_correct"]
        except Exception:
            # Fallback to direct generation if token extraction encounters vocabulary misalignment
            color_a, shape_a = obj_a
            color_b, shape_b = obj_b
            prompt = (
                f"Question: Is the {color_a} {shape_a} {relation} the {color_b} {shape_b}? "
                "Answer with only 'Yes' or 'No'."
            )
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt_text, images=[image], return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)

            generated_texts = self.processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            response = generated_texts[0].strip().lower()
            return "yes" in response and "no" not in response

    def evaluate_dataset_at_severity(self, dataset: List[Dict[str, Any]], severity: int) -> Dict[str, Any]:
        """Runs the generative evaluation across a degraded dataset split with live progress."""
        correct = 0
        total = len(dataset)

        for idx, sample in enumerate(dataset, start=1):
            degraded_img = PhotometricDegradationEngine.apply_pipeline(
                sample["image"], severity=severity
            )
            labels = sample.get("labels", {"obj_a": ("blue", "square"), "obj_b": ("red", "square")})
            is_correct = self.evaluate_sample(
                degraded_img,
                sample["target_relation"],
                labels["obj_a"],
                labels["obj_b"]
            )
            if is_correct:
                correct += 1

        accuracy = (correct / total) * 100.0 if total else 0.0
        return {
            "severity_level": severity,
            "total_samples": total,
            "accuracy": round(accuracy, 2)
        }