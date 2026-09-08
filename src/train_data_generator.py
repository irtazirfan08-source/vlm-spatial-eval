import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.spatial_generator import SpatialDatasetGenerator


OPPOSITE_RELATIONS = {
    "above": "below",
    "below": "above",
    "to the left of": "to the right of",
    "to the right of": "to the left of",
}


class SpatialInstructionDatasetBuilder:
    """Generates balanced multimodal conversational instruction pairs for spatial VLM alignment."""

    def __init__(self, output_dir: str = "data/fine_tune", canvas_size: int = 224):
        self.output_dir = Path(output_dir)
        self.canvas_size = canvas_size
        self.generator = SpatialDatasetGenerator(canvas_size=canvas_size)

    def _build_conversation(
        self, prompt_text: str, answer_text: str
    ) -> List[Dict[str, Any]]:
        """Formats the query into the chat format expected by SmolVLM."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer_text},
                ],
            },
        ]

    def build_split(
        self, split_name: str, num_samples_per_relation: int
    ) -> List[Dict[str, Any]]:
        """Generates raw synthetic samples and pairs each with positive and negative queries."""
        split_dir = self.output_dir / split_name
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        raw_samples = self.generator.generate_benchmark_split(
            num_samples_per_relation=num_samples_per_relation
        )
        records = []
        record_idx = 0

        for raw in raw_samples:
            img: Image.Image = raw["image"]
            relation = raw["target_relation"]
            color_a, shape_a = raw["labels"]["obj_a"]
            color_b, shape_b = raw["labels"]["obj_b"]

            # Save the image once per visual scene
            image_filename = f"{split_name}_{record_idx:05d}.png"
            image_filepath = images_dir / image_filename
            img.save(image_filepath, format="PNG")

            # 1. Positive Prompt (Ground Truth: 'Yes')
            pos_prompt = (
                f"Question: Is the {color_a} {shape_a} {relation} the {color_b} {shape_b}? "
                "Answer with only 'Yes' or 'No'."
            )
            records.append(
                {
                    "id": f"{split_name}_{record_idx:05d}_pos",
                    "image_path": str(image_filepath),
                    "relation": relation,
                    "conversations": self._build_conversation(pos_prompt, "Yes"),
                    "ground_truth": "Yes",
                }
            )

            # 2. Negative Counterfactual Prompt (Ground Truth: 'No')
            # Flip the directional preposition to create a hard negative foil
            neg_relation = OPPOSITE_RELATIONS[relation]
            neg_prompt = (
                f"Question: Is the {color_a} {shape_a} {neg_relation} the {color_b} {shape_b}? "
                "Answer with only 'Yes' or 'No'."
            )
            records.append(
                {
                    "id": f"{split_name}_{record_idx:05d}_neg",
                    "image_path": str(image_filepath),
                    "relation": neg_relation,
                    "conversations": self._build_conversation(neg_prompt, "No"),
                    "ground_truth": "No",
                }
            )

            record_idx += 1

        # Shuffle records to prevent ordering bias during gradient descent
        random.seed(42)
        random.shuffle(records)

        metadata_file = split_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        print(
            f"[{split_name.upper()}] Generated {len(records)} balanced records ({len(raw_samples)} images) -> {metadata_file}"
        )
        return records

    def generate_all(
        self, train_samples_per_rel: int = 50, val_samples_per_rel: int = 10
    ):
        """Builds both training (400 queries) and validation (80 queries) sets."""
        print("--> Generating instruction fine-tuning dataset...")
        train_records = self.build_split("train", train_samples_per_rel)
        val_records = self.build_split("val", val_samples_per_rel)
        print("--> Dataset generation complete.\n")
        return train_records, val_records


if __name__ == "__main__":
    builder = SpatialInstructionDatasetBuilder(output_dir="data/fine_tune")
    # 50 samples * 4 relations * 2 queries (Yes/No) = 400 training samples
    # 10 samples * 4 relations * 2 queries (Yes/No) = 80 validation samples
    builder.generate_all(train_samples_per_rel=50, val_samples_per_rel=10)