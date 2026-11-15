"""Counterfactual mirror symmetry and depth-occlusion evaluation for VLMs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw

BBoxLike = Union[Tuple[float, float, float, float], Sequence[float], Any]

HORIZONTAL_INVERSIONS: Dict[str, str] = {
    "left_of": "right_of",
    "right_of": "left_of",
}

INVARIANT_RELATIONS = {"above", "below", "inside", "in_front_of", "behind"}


def extract_bbox_coords(bbox: BBoxLike) -> Tuple[float, float, float, float]:
    """Extract (x1, y1, x2, y2) from a tuple/list or BoundingBox dataclass."""
    if hasattr(bbox, "x1") and hasattr(bbox, "y1") and hasattr(bbox, "x2") and hasattr(bbox, "y2"):
        return (float(bbox.x1), float(bbox.y1), float(bbox.x2), float(bbox.y2))
    if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    raise ValueError(f"Unsupported bounding box format: {bbox!r}")


def flip_bbox_horizontally(
    bbox: BBoxLike, image_width: int
) -> Tuple[float, float, float, float]:
    """Mirror bounding box coordinates across x = image_width / 2."""
    if image_width <= 0:
        raise ValueError("image_width must be positive.")
    x1, y1, x2, y2 = extract_bbox_coords(bbox)
    flipped_x1 = float(image_width) - x2
    flipped_x2 = float(image_width) - x1
    return (flipped_x1, y1, flipped_x2, y2)


def invert_horizontal_relation(relation: str) -> str:
    """Invert left/right predicates under horizontal reflection while preserving invariants."""
    rel = relation.strip().lower()
    if rel in HORIZONTAL_INVERSIONS:
        return HORIZONTAL_INVERSIONS[rel]
    if rel in INVARIANT_RELATIONS:
        return rel
    raise ValueError(f"Unknown spatial relation: {relation!r}")


def compute_bbox_iou(bbox_a: BBoxLike, bbox_b: BBoxLike) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes."""
    ax1, ay1, ax2, ay2 = extract_bbox_coords(bbox_a)
    bx1, by1, bx2, by2 = extract_bbox_coords(bbox_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0.0:
        return 0.0
    return inter_area / union


def compute_occlusion_ratio(
    foreground_bbox: BBoxLike, background_bbox: BBoxLike
) -> float:
    """Compute fraction of background_bbox area occluded by foreground_bbox."""
    fx1, fy1, fx2, fy2 = extract_bbox_coords(foreground_bbox)
    bx1, by1, bx2, by2 = extract_bbox_coords(background_bbox)

    inter_w = max(0.0, min(fx2, bx2) - max(fx1, bx1))
    inter_h = max(0.0, min(fy2, by2) - max(fy1, by1))
    inter_area = inter_w * inter_h

    bg_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if bg_area <= 0.0:
        return 0.0
    return inter_area / bg_area


def evaluate_occlusion_relation(
    subject_bbox: BBoxLike,
    subject_z: int,
    object_bbox: BBoxLike,
    object_z: int,
    min_iou: float = 0.05,
) -> str:
    """Determine depth predicate ('in_front_of', 'behind', or 'disjoint') from overlap and z-order."""
    iou = compute_bbox_iou(subject_bbox, object_bbox)
    if iou < min_iou or subject_z == object_z:
        return "disjoint"
    return "in_front_of" if subject_z > object_z else "behind"


@dataclass
class CounterfactualFlipPair:
    """Paired original and horizontally mirrored spatial evaluation sample."""

    original_image: Image.Image
    flipped_image: Image.Image
    subject_label: str
    object_label: str
    original_relation: str
    flipped_relation: str
    original_subject_bbox: Tuple[float, float, float, float]
    original_object_bbox: Tuple[float, float, float, float]
    flipped_subject_bbox: Tuple[float, float, float, float]
    flipped_object_bbox: Tuple[float, float, float, float]


def generate_counterfactual_flip_pair(
    image: Image.Image,
    subject_bbox: BBoxLike,
    object_bbox: BBoxLike,
    relation: str,
    subject_label: str = "red square",
    object_label: str = "blue circle",
) -> CounterfactualFlipPair:
    """Create a horizontally mirrored counterfactual test pair with updated bounding boxes."""
    width, _ = image.size
    flipped_img = image.transpose(Image.FLIP_LEFT_RIGHT)

    orig_subj = extract_bbox_coords(subject_bbox)
    orig_obj = extract_bbox_coords(object_bbox)

    return CounterfactualFlipPair(
        original_image=image,
        flipped_image=flipped_img,
        subject_label=subject_label,
        object_label=object_label,
        original_relation=relation,
        flipped_relation=invert_horizontal_relation(relation),
        original_subject_bbox=orig_subj,
        original_object_bbox=orig_obj,
        flipped_subject_bbox=flip_bbox_horizontally(orig_subj, width),
        flipped_object_bbox=flip_bbox_horizontally(orig_obj, width),
    )


@dataclass
class OcclusionSample:
    """Synthetic depth-ordered scene sample for occlusion predicate benchmarking."""

    image: Image.Image
    subject_label: str
    object_label: str
    subject_bbox: Tuple[float, float, float, float]
    object_bbox: Tuple[float, float, float, float]
    subject_z: int
    object_z: int
    relation: str
    occlusion_ratio: float


class OcclusionSceneRenderer:
    """Renders z-ordered overlapping 2D shapes for 'in_front_of' and 'behind' evaluation."""

    def __init__(self, image_size: int = 224, bg_color: Tuple[int, int, int] = (245, 245, 245)):
        self.image_size = image_size
        self.bg_color = bg_color

    def render_pair(
        self,
        subject_in_front: bool = True,
        subject_color: Tuple[int, int, int] = (220, 50, 47),
        object_color: Tuple[int, int, int] = (38, 139, 210),
        subject_label: str = "red box",
        object_label: str = "blue box",
    ) -> OcclusionSample:
        """Render an overlapping two-object scene with explicit depth z-ordering."""
        img = Image.new("RGB", (self.image_size, self.image_size), self.bg_color)
        draw = ImageDraw.Draw(img)

        # Overlapping bounding boxes near image center
        s = self.image_size
        subject_bbox = (int(0.28 * s), int(0.30 * s), int(0.58 * s), int(0.62 * s))
        object_bbox = (int(0.42 * s), int(0.38 * s), int(0.72 * s), int(0.70 * s))

        subject_z = 2 if subject_in_front else 1
        object_z = 1 if subject_in_front else 2

        layers = [
            (subject_z, subject_bbox, subject_color),
            (object_z, object_bbox, object_color),
        ]
        layers.sort(key=lambda item: item[0])

        for _, bbox, color in layers:
            draw.rectangle(bbox, fill=color, outline=(25, 25, 25), width=2)

        relation = evaluate_occlusion_relation(
            subject_bbox, subject_z, object_bbox, object_z
        )
        fg_bbox = subject_bbox if subject_in_front else object_bbox
        bg_bbox = object_bbox if subject_in_front else subject_bbox
        occ_ratio = compute_occlusion_ratio(fg_bbox, bg_bbox)

        return OcclusionSample(
            image=img,
            subject_label=subject_label,
            object_label=object_label,
            subject_bbox=tuple(float(v) for v in subject_bbox),
            object_bbox=tuple(float(v) for v in object_bbox),
            subject_z=subject_z,
            object_z=object_z,
            relation=relation,
            occlusion_ratio=occ_ratio,
        )


@dataclass
class SymmetryAuditReport:
    """Aggregate counterfactual mirror-symmetry diagnostics."""

    total_pairs: int
    original_accuracy: float
    counterfactual_accuracy: float
    symmetry_consistency: float
    directional_bias: float


class CounterfactualSymmetryAuditor:
    """Audits whether a VLM scoring function respects horizontal mirror symmetry."""

    def __init__(
        self,
        scorer_fn: Callable[[Image.Image, str, str], bool],
    ):
        """
        Args:
            scorer_fn: Callable taking (image, positive_prompt, negative_prompt) and
                returning True if positive_prompt scores higher than negative_prompt.
        """
        self.scorer_fn = scorer_fn

    @staticmethod
    def _format_prompt(subject: str, relation: str, obj: str) -> str:
        readable_rel = relation.replace("_", " ")
        return f"The {subject} is {readable_rel} the {obj}."

    def evaluate_pairs(self, pairs: Sequence[CounterfactualFlipPair]) -> SymmetryAuditReport:
        if not pairs:
            raise ValueError("pairs must be non-empty.")

        orig_correct = 0
        flip_correct = 0
        both_consistent = 0
        left_predictions = 0
        right_predictions = 0

        for pair in pairs:
            orig_pos = self._format_prompt(
                pair.subject_label, pair.original_relation, pair.object_label
            )
            orig_neg = self._format_prompt(
                pair.subject_label, pair.flipped_relation, pair.object_label
            )
            pred_orig_ok = bool(self.scorer_fn(pair.original_image, orig_pos, orig_neg))

            flip_pos = self._format_prompt(
                pair.subject_label, pair.flipped_relation, pair.object_label
            )
            flip_neg = self._format_prompt(
                pair.subject_label, pair.original_relation, pair.object_label
            )
            pred_flip_ok = bool(self.scorer_fn(pair.flipped_image, flip_pos, flip_neg))

            if pred_orig_ok:
                orig_correct += 1
            if pred_flip_ok:
                flip_correct += 1
            if pred_orig_ok and pred_flip_ok:
                both_consistent += 1

            # Track predicted horizontal direction across both views
            for rel, is_correct in [
                (pair.original_relation, pred_orig_ok),
                (pair.flipped_relation, pred_flip_ok),
            ]:
                predicted_rel = rel if is_correct else invert_horizontal_relation(rel)
                if predicted_rel == "left_of":
                    left_predictions += 1
                elif predicted_rel == "right_of":
                    right_predictions += 1

        n = len(pairs)
        total_preds = max(1, left_predictions + right_predictions)
        bias = abs(left_predictions - right_predictions) / total_preds

        return SymmetryAuditReport(
            total_pairs=n,
            original_accuracy=orig_correct / n,
            counterfactual_accuracy=flip_correct / n,
            symmetry_consistency=both_consistent / n,
            directional_bias=bias,
        )


def _demo_cli() -> None:
    """Run a self-contained CLI demonstration of counterfactual symmetry & occlusion audits."""
    pairs: List[CounterfactualFlipPair] = []
    for idx in range(4):
        img = Image.new("RGB", (224, 224), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        if idx % 2 == 0:
            subj_box, obj_box, rel = (30, 90, 70, 130), (150, 90, 190, 130), "left_of"
        else:
            subj_box, obj_box, rel = (150, 90, 190, 130), (30, 90, 70, 130), "right_of"
        draw.rectangle(subj_box, fill=(220, 50, 47))
        draw.rectangle(obj_box, fill=(38, 139, 210))
        pairs.append(generate_counterfactual_flip_pair(img, subj_box, obj_box, rel))

    # 1. Spatially grounded scorer (checks red pixel centroid vs blue pixel centroid)
    def grounded_scorer(image: Image.Image, pos_prompt: str, _neg_prompt: str) -> bool:
        arr = np.asarray(image)
        red_x = np.where(arr[:, :, 0] > 180)[1].mean()
        blue_x = np.where(arr[:, :, 2] > 180)[1].mean()
        actual_rel = "left of" if red_x < blue_x else "right of"
        return actual_rel in pos_prompt

    # 2. Language-prior biased scorer (always prefers 'left of' regardless of visual flip)
    def biased_scorer(_image: Image.Image, pos_prompt: str, _neg_prompt: str) -> bool:
        return "left of" in pos_prompt

    grounded_report = CounterfactualSymmetryAuditor(grounded_scorer).evaluate_pairs(pairs)
    biased_report = CounterfactualSymmetryAuditor(biased_scorer).evaluate_pairs(pairs)

    renderer = OcclusionSceneRenderer()
    front_sample = renderer.render_pair(subject_in_front=True)
    behind_sample = renderer.render_pair(subject_in_front=False)

    print("=========================================================================")
    print(" VLM-SpatialEval: Counterfactual Symmetry & Occlusion Diagnostic Audit")
    print("=========================================================================")
    print("| Evaluator Profile         | Orig Acc | Flip Acc | Symmetry Cons. | Bias |")
    print("|:--------------------------|:---------|:---------|:---------------|:-----|")
    print(
        f"| Visually Grounded Oracle  | {grounded_report.original_accuracy*100:6.1f}%  | "
        f"{grounded_report.counterfactual_accuracy*100:6.1f}%  | "
        f"{grounded_report.symmetry_consistency*100:12.1f}%  | "
        f"{grounded_report.directional_bias:4.2f} |"
    )
    print(
        f"| Language-Prior Biased VLM | {biased_report.original_accuracy*100:6.1f}%  | "
        f"{biased_report.counterfactual_accuracy*100:6.1f}%  | "
        f"{biased_report.symmetry_consistency*100:12.1f}%  | "
        f"{biased_report.directional_bias:4.2f} |"
    )
    print("-------------------------------------------------------------------------")
    print(
        f"Occlusion Synthesis Check -> Sample 1: '{front_sample.relation}' "
        f"(overlap={front_sample.occlusion_ratio:.2f}) | "
        f"Sample 2: '{behind_sample.relation}' (overlap={behind_sample.occlusion_ratio:.2f})"
    )


if __name__ == "__main__":
    _demo_cli()