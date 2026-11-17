"""Confidence calibration metrics (ECE, MCE, Overconfidence, and Brier Score) for VLMs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


def margin_to_probability(margin: float, temperature: float = 1.0) -> float:
    """Convert a pairwise logit or similarity margin into a probability in [0, 1]."""
    if temperature <= 0.0:
        raise ValueError("temperature must be strictly positive.")
    scaled = max(-60.0, min(60.0, float(margin) / float(temperature)))
    return 1.0 / (1.0 + math.exp(-scaled))


def margin_to_confidence(margin: float, temperature: float = 1.0) -> float:
    """Convert a pairwise margin into top-1 prediction confidence in [0.5, 1.0]."""
    prob = margin_to_probability(margin, temperature=temperature)
    return max(prob, 1.0 - prob)


def compute_brier_score(
    confidences: Sequence[float],
    correctness: Sequence[Union[bool, int, float]],
) -> float:
    """Compute Brier Score: mean squared error between predicted confidence and correctness."""
    if len(confidences) != len(correctness):
        raise ValueError("confidences and correctness must have identical lengths.")
    if not confidences:
        raise ValueError("confidences must be non-empty.")

    total_sq_error = 0.0
    for conf, corr in zip(confidences, correctness):
        c_val = float(conf)
        y_val = float(corr)
        if not (0.0 <= c_val <= 1.0):
            raise ValueError(f"Confidence value {c_val} must lie within [0.0, 1.0].")
        if not (0.0 <= y_val <= 1.0):
            raise ValueError(f"Correctness value {y_val} must lie within [0.0, 1.0].")
        total_sq_error += (c_val - y_val) ** 2

    return total_sq_error / len(confidences)


@dataclass
class CalibrationBin:
    """Statistics for a single equal-width confidence histogram bin."""

    bin_index: int
    lower_bound: float
    upper_bound: float
    count: int
    mean_confidence: float
    empirical_accuracy: float
    calibration_gap: float


@dataclass
class CalibrationSummary:
    """Aggregate confidence calibration metrics for a set of VLM predictions."""

    num_samples: int
    accuracy: float
    mean_confidence: float
    ece: float
    mce: float
    overconfidence_error: float
    brier_score: float
    bins: List[CalibrationBin]


def compute_expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[Union[bool, int, float]],
    n_bins: int = 10,
) -> CalibrationSummary:
    """Compute ECE, MCE, Overconfidence Error, and Brier Score across equal-width bins."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")
    if len(confidences) != len(correctness):
        raise ValueError("confidences and correctness must have identical lengths.")
    if not confidences:
        raise ValueError("confidences must be non-empty.")

    brier = compute_brier_score(confidences, correctness)
    n_total = len(confidences)

    bin_confs: List[List[float]] = [[] for _ in range(n_bins)]
    bin_corrs: List[List[float]] = [[] for _ in range(n_bins)]

    for conf, corr in zip(confidences, correctness):
        c_val = float(conf)
        y_val = float(corr)
        idx = min(int(c_val * n_bins), n_bins - 1)
        bin_confs[idx].append(c_val)
        bin_corrs[idx].append(y_val)

    ece = 0.0
    mce = 0.0
    overconfidence = 0.0
    bin_summaries: List[CalibrationBin] = []

    for idx in range(n_bins):
        lower = idx / n_bins
        upper = (idx + 1) / n_bins
        count = len(bin_confs[idx])
        if count == 0:
            bin_summaries.append(
                CalibrationBin(
                    bin_index=idx,
                    lower_bound=round(lower, 4),
                    upper_bound=round(upper, 4),
                    count=0,
                    mean_confidence=0.0,
                    empirical_accuracy=0.0,
                    calibration_gap=0.0,
                )
            )
            continue

        mean_conf = sum(bin_confs[idx]) / count
        emp_acc = sum(bin_corrs[idx]) / count
        gap = abs(emp_acc - mean_conf)
        weight = count / n_total

        ece += weight * gap
        mce = max(mce, gap)
        overconfidence += weight * max(0.0, mean_conf - emp_acc)

        bin_summaries.append(
            CalibrationBin(
                bin_index=idx,
                lower_bound=round(lower, 4),
                upper_bound=round(upper, 4),
                count=count,
                mean_confidence=round(mean_conf, 4),
                empirical_accuracy=round(emp_acc, 4),
                calibration_gap=round(gap, 4),
            )
        )

    overall_acc = sum(float(y) for y in correctness) / n_total
    overall_conf = sum(float(c) for c in confidences) / n_total

    return CalibrationSummary(
        num_samples=n_total,
        accuracy=round(overall_acc, 4),
        mean_confidence=round(overall_conf, 4),
        ece=round(ece, 4),
        mce=round(mce, 4),
        overconfidence_error=round(overconfidence, 4),
        brier_score=round(brier, 4),
        bins=bin_summaries,
    )


class CalibrationTracker:
    """Tracks VLM confidence calibration across photometric degradation severity levels."""

    def __init__(self, n_bins: int = 10):
        if n_bins < 1:
            raise ValueError("n_bins must be at least 1.")
        self.n_bins = n_bins
        self._records: Dict[int, List[Dict[str, float]]] = {}

    def add_prediction(
        self,
        severity: int,
        is_correct: Union[bool, int],
        confidence: Optional[float] = None,
        margin: Optional[float] = None,
        temperature: float = 1.0,
    ) -> None:
        """Record a single prediction using either explicit confidence or raw logit margin."""
        if confidence is None and margin is None:
            raise ValueError("Provide either confidence or margin.")
        resolved_conf = (
            float(confidence)
            if confidence is not None
            else margin_to_confidence(float(margin), temperature=temperature)
        )
        if not (0.0 <= resolved_conf <= 1.0):
            raise ValueError(f"Resolved confidence {resolved_conf} must be in [0.0, 1.0].")

        self._records.setdefault(int(severity), []).append(
            {"confidence": resolved_conf, "correct": float(bool(is_correct))}
        )

    def summarize_by_severity(self) -> Dict[int, CalibrationSummary]:
        """Compute CalibrationSummary independently for each recorded severity tier."""
        summaries: Dict[int, CalibrationSummary] = {}
        for sev in sorted(self._records.keys()):
            items = self._records[sev]
            confs = [row["confidence"] for row in items]
            corrs = [row["correct"] for row in items]
            summaries[sev] = compute_expected_calibration_error(
                confs, corrs, n_bins=self.n_bins
            )
        return summaries

    def to_markdown_table(self) -> str:
        """Render a GitHub-Flavored Markdown table of calibration metrics across severities."""
        summaries = self.summarize_by_severity()
        lines = [
            "| Severity | Samples | Accuracy | Mean Conf. | ECE ↓ | MCE ↓ | Overconf. ↓ | Brier ↓ |",
            "|:---|:---|:---|:---|:---|:---|:---|:---|",
        ]
        for sev, s in summaries.items():
            lines.append(
                f"| L{sev} | {s.num_samples} | {s.accuracy * 100:.1f}% | "
                f"{s.mean_confidence:.3f} | {s.ece:.4f} | {s.mce:.4f} | "
                f"{s.overconfidence_error:.4f} | {s.brier_score:.4f} |"
            )
        return "\n".join(lines)

    def to_latex_table(self) -> str:
        """Render an academic LaTeX booktabs table of calibration degradation."""
        summaries = self.summarize_by_severity()
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{VLM Spatial Confidence Calibration Across Photometric Degradation Tiers}",
            r"\label{tab:vlm_calibration}",
            r"\begin{tabular}{lccccccc}",
            r"\toprule",
            r"Severity & $N$ & Acc. (\%) & Mean Conf. & ECE $\downarrow$ & MCE $\downarrow$ & Overconf. $\downarrow$ & Brier $\downarrow$ \\",
            r"\midrule",
        ]
        for sev, s in summaries.items():
            lines.append(
                f"L{sev} & {s.num_samples} & {s.accuracy * 100:.1f} & "
                f"{s.mean_confidence:.3f} & {s.ece:.4f} & {s.mce:.4f} & "
                f"{s.overconfidence_error:.4f} & {s.brier_score:.4f} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)

    def export_json(self, output_path: str) -> None:
        """Serialize per-severity calibration summaries to disk as JSON."""
        summaries = self.summarize_by_severity()
        payload: Dict[str, Any] = {
            f"L{sev}": asdict(summary) for sev, summary in summaries.items()
        }
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _demo_cli() -> None:
    """Demonstrate calibration collapse when VLM accuracy drops under photometric noise."""
    tracker = CalibrationTracker(n_bins=5)

    # Simulated VLM behavior: stays ~84% confident even as accuracy degrades from 90% to 40%
    simulated_tiers = {
        0: [(0.86, True)] * 9 + [(0.80, False)] * 1,
        1: [(0.85, True)] * 7 + [(0.82, False)] * 3,
        2: [(0.84, True)] * 5 + [(0.83, False)] * 5,
        3: [(0.83, True)] * 4 + [(0.84, False)] * 6,
    }
    for sev, rows in simulated_tiers.items():
        for conf, corr in rows:
            tracker.add_prediction(severity=sev, is_correct=corr, confidence=conf)

    print("=============================================================================")
    print(" VLM-SpatialEval: Confidence Calibration & Overconfidence Degradation Audit")
    print("=============================================================================")
    print(tracker.to_markdown_table())


if __name__ == "__main__":
    _demo_cli()