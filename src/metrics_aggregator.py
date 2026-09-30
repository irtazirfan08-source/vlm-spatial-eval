import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class EvaluationResult:
    sample_id: str
    relation: str
    predicted: str
    ground_truth: str
    is_correct: bool
    is_perturbed: bool = False
    perturbation_type: Optional[str] = None


class SpatialMetricsAggregator:
    """Aggregates spatial reasoning evaluation outputs and computes

    robustness degradation metrics across relation categories and visual perturbations.
    """

    def __init__(self):
        self.results: List[EvaluationResult] = []

    def add_evaluation(
        self,
        sample_id: str,
        relation: str,
        predicted: str,
        ground_truth: str,
        is_perturbed: bool = False,
        perturbation_type: Optional[str] = None,
    ) -> None:
        self.results.append(
            EvaluationResult(
                sample_id=sample_id,
                relation=relation.lower().strip(),
                predicted=str(predicted).strip(),
                ground_truth=str(ground_truth).strip(),
                is_correct=(str(predicted).strip().lower() == str(ground_truth).strip().lower()),
                is_perturbed=is_perturbed,
                perturbation_type=perturbation_type,
            )
        )

    def get_clean_accuracy(self) -> float:
        clean = [r for r in self.results if not r.is_perturbed]
        if not clean:
            return 0.0
        return sum(1 for r in clean if r.is_correct) / len(clean)

    def get_perturbed_accuracy(self, perturbation_type: Optional[str] = None) -> float:
        if perturbation_type:
            perturbed = [
                r for r in self.results
                if r.is_perturbed and r.perturbation_type == perturbation_type
            ]
        else:
            perturbed = [r for r in self.results if r.is_perturbed]

        if not perturbed:
            return 0.0
        return sum(1 for r in perturbed if r.is_correct) / len(perturbed)

    def compute_robustness_drop(self, perturbation_type: Optional[str] = None) -> float:
        """Calculates accuracy degradation rate: (Clean - Perturbed) / Clean."""
        clean_acc = self.get_clean_accuracy()
        if clean_acc == 0.0:
            return 0.0
        pert_acc = self.get_perturbed_accuracy(perturbation_type)
        return max(0.0, (clean_acc - pert_acc) / clean_acc)

    def compute_per_relation_breakdown(self) -> Dict[str, Dict[str, float]]:
        relations = sorted(list({r.relation for r in self.results}))
        breakdown: Dict[str, Dict[str, float]] = {}

        for rel in relations:
            rel_clean = [r for r in self.results if r.relation == rel and not r.is_perturbed]
            rel_pert = [r for r in self.results if r.relation == rel and r.is_perturbed]

            clean_acc = (
                sum(1 for r in rel_clean if r.is_correct) / len(rel_clean)
                if rel_clean else 0.0
            )
            pert_acc = (
                sum(1 for r in rel_pert if r.is_correct) / len(rel_pert)
                if rel_pert else 0.0
            )
            drop = (
                max(0.0, (clean_acc - pert_acc) / clean_acc)
                if clean_acc > 0.0 else 0.0
            )

            breakdown[rel] = {
                "clean_samples": len(rel_clean),
                "clean_accuracy": round(clean_acc, 4),
                "perturbed_samples": len(rel_pert),
                "perturbed_accuracy": round(pert_acc, 4),
                "relative_drop": round(drop, 4),
            }

        return breakdown

    def generate_markdown_report(self) -> str:
        breakdown = self.compute_per_relation_breakdown()
        overall_clean = self.get_clean_accuracy()
        overall_pert = self.get_perturbed_accuracy()
        overall_drop = self.compute_robustness_drop()

        lines = [
            "### Spatial Reasoning & Robustness Benchmark Summary",
            "",
            f"- **Overall Clean Accuracy**: {overall_clean * 100:.2f}%",
            f"- **Overall Perturbed Accuracy**: {overall_pert * 100:.2f}%",
            f"- **Relative Robustness Drop**: {overall_drop * 100:.2f}%",
            "",
            "| Relation | Clean Samples | Clean Acc | Perturbed Samples | Perturbed Acc | Degradation Drop |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]

        for rel, stats in breakdown.items():
            lines.append(
                f"| `{rel}` | {stats['clean_samples']} | {stats['clean_accuracy']*100:.1f}% | "
                f"{stats['perturbed_samples']} | {stats['perturbed_accuracy']*100:.1f}% | "
                f"{stats['relative_drop']*100:.1f}% |"
            )

        return "\n".join(lines)

    def export_json(self, output_path: str) -> None:
        data = {
            "summary": {
                "total_samples": len(self.results),
                "clean_accuracy": round(self.get_clean_accuracy(), 4),
                "perturbed_accuracy": round(self.get_perturbed_accuracy(), 4),
                "relative_drop": round(self.compute_robustness_drop(), 4),
            },
            "per_relation": self.compute_per_relation_breakdown(),
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)