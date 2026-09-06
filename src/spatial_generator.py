import random
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw

RELATION_INVERSES: Dict[str, str] = {
    "to the left of": "to the right of",
    "to the right of": "to the left of",
    "above": "below",
    "below": "above",
}

SHAPES: List[str] = ["circle", "square", "triangle"]
COLORS: Dict[str, Tuple[int, int, int]] = {
    "red": (220, 20, 60),
    "blue": (30, 144, 255),
    "green": (46, 139, 87),
    "yellow": (255, 215, 0),
}


class SpatialDatasetGenerator:
    """Generates synthetic multi-object scenes paired with contrastive relational prompts."""

    def __init__(self, canvas_size: int = 384):
        self.canvas_size = canvas_size

    def _draw_shape(
        self,
        draw: ImageDraw.ImageDraw,
        shape: str,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
    ) -> None:
        x1, y1, x2, y2 = bbox
        if shape == "circle":
            draw.ellipse([x1, y1, x2, y2], fill=color)
        elif shape == "square":
            draw.rectangle([x1, y1, x2, y2], fill=color)
        elif shape == "triangle":
            draw.polygon([(x1, y2), ((x1 + x2) // 2, y1), (x2, y2)], fill=color)

    def generate_directional_sample(self, relation: str = "to the left of") -> Dict[str, Any]:
        """Renders two non-overlapping shapes satisfying a given directional relation."""
        image = Image.new("RGB", (self.canvas_size, self.canvas_size), color=(240, 240, 240))
        draw = ImageDraw.Draw(image)

        color_a, color_b = random.sample(list(COLORS.keys()), 2)
        shape_a, shape_b = random.sample(SHAPES, 2)
        obj_size = 60

        if relation == "to the left of":
            box_a = (40, 160, 40 + obj_size, 160 + obj_size)
            box_b = (260, 160, 260 + obj_size, 160 + obj_size)
        elif relation == "to the right of":
            box_a = (260, 160, 260 + obj_size, 160 + obj_size)
            box_b = (40, 160, 40 + obj_size, 160 + obj_size)
        elif relation == "above":
            box_a = (160, 40, 160 + obj_size, 40 + obj_size)
            box_b = (160, 260, 160 + obj_size, 260 + obj_size)
        elif relation == "below":
            box_a = (160, 260, 160 + obj_size, 260 + obj_size)
            box_b = (160, 40, 160 + obj_size, 40 + obj_size)
        else:
            raise ValueError(f"Unsupported spatial relation: {relation}")

        self._draw_shape(draw, shape_a, box_a, COLORS[color_a])
        self._draw_shape(draw, shape_b, box_b, COLORS[color_b])

        positive_text = f"a {color_a} {shape_a} {relation} a {color_b} {shape_b}"
        foil_relation = RELATION_INVERSES[relation]
        negative_text = f"a {color_a} {shape_a} {foil_relation} a {color_b} {shape_b}"

        return {
            "image": image,
            "target_relation": relation,
            "positive_prompt": positive_text,
            "negative_prompt": negative_text,
            "labels": {"obj_a": (color_a, shape_a), "obj_b": (color_b, shape_b)},
        }

    def generate_benchmark_split(self, num_samples_per_relation: int = 15) -> List[Dict[str, Any]]:
        """Produces a balanced evaluation split across all canonical spatial relations."""
        dataset: List[Dict[str, Any]] = []
        for rel in RELATION_INVERSES.keys():
            for _ in range(num_samples_per_relation):
                dataset.append(self.generate_directional_sample(relation=rel))
        return dataset