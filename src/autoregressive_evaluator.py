from typing import Any, Dict, List
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

from src.perturbations import PhotometricDegradationEngine


class AutoregressiveSpatialEvaluator:
    """Evaluates generative autoregressive VLMs on directional spatial discrimination."""

    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.bfloat16,
        ).to(self.device).eval()

    def evaluate_sample(self, image: Image.Image, relation: str, obj_a: tuple, obj_b: tuple) -> bool:
        """
        Prompts the generative VLM with a binary question:
        'Is the {color_a} {shape_a} {relation} the {color_b} {shape_b}? Answer Yes or No.'
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
            generated_ids = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)

        generated_texts = self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        response = generated_texts[0].strip().lower()

        # The true directional relation holds, so 'yes' is the correct answer
        return "yes" in response and "no" not in response

    def evaluate_dataset_at_severity(self, dataset: List[Dict[str, Any]], severity: int) -> Dict[str, Any]:
        """Runs the generative evaluation across a degraded dataset split with live progress."""
        correct = 0
        total = len(dataset)

        for idx, sample in enumerate(dataset, start=1):
            print(f"   --> [L_{severity}] Sample {idx}/{total} processing...", flush=True)
            degraded_img = PhotometricDegradationEngine.apply_pipeline(
                sample["image"], severity=severity
            )
            is_correct = self.evaluate_sample(
                degraded_img,
                sample["target_relation"],
                sample["labels"]["obj_a"],
                sample["labels"]["obj_b"]
            )
            if is_correct:
                correct += 1

        accuracy = (correct / total) * 100.0 if total else 0.0
        return {
            "severity_level": severity,
            "total_samples": total,
            "accuracy": round(accuracy, 2)
        }