"""Relative directional and topological spatial relation evaluator for VLMs."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SpatialRelation(str, Enum):
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    CONTAINS = "contains"
    INSIDE = "inside"
    OVERLAPPING = "overlapping"
    DISJOINT = "disjoint"


@dataclass
class BoundingBox:
    """Represents a 2D bounding box normalized or in absolute pixel coordinates [xmin, ymin, xmax, ymax]."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        if self.xmin > self.xmax or self.ymin > self.ymax:
            raise ValueError(
                f"Invalid coordinates: xmin ({self.xmin}) must be <= xmax ({self.xmax}) and ymin ({self.ymin}) <= ymax ({self.ymax})"
            )

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        """Calculates (center_x, center_y)."""
        return (self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0


class DirectionalEvaluator:
    """Evaluates relative spatial, topological, and orientation relations between bounding boxes."""

    @staticmethod
    def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
        """Calculates Intersection over Union (IoU) between two bounding boxes."""
        inter_xmin = max(box_a.xmin, box_b.xmin)
        inter_ymin = max(box_a.ymin, box_b.ymin)
        inter_xmax = min(box_a.xmax, box_b.xmax)
        inter_ymax = min(box_a.ymax, box_b.ymax)

        inter_width = max(0.0, inter_xmax - inter_xmin)
        inter_height = max(0.0, inter_ymax - inter_ymin)
        intersection_area = inter_width * inter_height

        union_area = box_a.area + box_b.area - intersection_area
        if union_area <= 0.0:
            return 0.0
        return intersection_area / union_area

    @staticmethod
    def evaluate_relative_directions(
        subject: BoundingBox, reference: BoundingBox, tolerance: float = 0.0
    ) -> Dict[str, bool]:
        """Evaluates whether subject satisfies directional predicates relative to reference in standard image coordinates (origin at top-left)."""
        subj_cx, subj_cy = subject.center
        ref_cx, ref_cy = reference.center

        return {
            "left_of": (subj_cx < ref_cx) and (subject.xmax <= reference.xmin + tolerance),
            "right_of": (subj_cx > ref_cx) and (subject.xmin >= reference.xmax - tolerance),
            "above": (subj_cy < ref_cy) and (subject.ymax <= reference.ymin + tolerance),
            "below": (subj_cy > ref_cy) and (subject.ymin >= reference.ymax - tolerance),
        }

    @staticmethod
    def evaluate_containment(inner: BoundingBox, outer: BoundingBox) -> bool:
        """Checks if inner bounding box is completely enclosed within outer bounding box."""
        return (
            inner.xmin >= outer.xmin
            and inner.xmax <= outer.xmax
            and inner.ymin >= outer.ymin
            and inner.ymax <= outer.ymax
        )

    def evaluate_pairwise_relation(
        self, subject: BoundingBox, reference: BoundingBox, tolerance: float = 0.0
    ) -> List[SpatialRelation]:
        """Classifies all satisfied spatial relations between subject and reference."""
        relations: List[SpatialRelation] = []

        if self.evaluate_containment(inner=subject, outer=reference):
            relations.append(SpatialRelation.INSIDE)
        if self.evaluate_containment(inner=reference, outer=subject):
            relations.append(SpatialRelation.CONTAINS)

        iou = self.compute_iou(subject, reference)
        if iou > 0.0:
            relations.append(SpatialRelation.OVERLAPPING)
        else:
            relations.append(SpatialRelation.DISJOINT)

        dirs = self.evaluate_relative_directions(subject, reference, tolerance=tolerance)
        if dirs["left_of"]:
            relations.append(SpatialRelation.LEFT_OF)
        if dirs["right_of"]:
            relations.append(SpatialRelation.RIGHT_OF)
        if dirs["above"]:
            relations.append(SpatialRelation.ABOVE)
        if dirs["below"]:
            relations.append(SpatialRelation.BELOW)

        return relations

    def evaluate_prediction_accuracy(
        self,
        predicted_pairs: List[Tuple[BoundingBox, BoundingBox]],
        ground_truth_relations: List[SpatialRelation],
    ) -> Dict[str, float]:
        """Computes multi-pair directional prediction accuracy and average IoU."""
        if not predicted_pairs or len(predicted_pairs) != len(ground_truth_relations):
            raise ValueError("Mismatched or empty predicted pairs and ground truth relations.")

        correct_count = 0
        total_iou = 0.0

        for (pred_sub, pred_ref), target_rel in zip(predicted_pairs, ground_truth_relations):
            detected_relations = self.evaluate_pairwise_relation(pred_sub, pred_ref)
            if target_rel in detected_relations:
                correct_count += 1
            total_iou += self.compute_iou(pred_sub, pred_ref)

        total = len(predicted_pairs)
        return {
            "accuracy": correct_count / total,
            "mean_pairwise_iou": total_iou / total,
            "total_evaluated": float(total),
        }