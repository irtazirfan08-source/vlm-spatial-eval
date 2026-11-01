# vlm-spatial-eval

[![CI Pipeline](https://github.com/irtazirfan08-source/vlm-spatial-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/irtazirfan08-source/vlm-spatial-eval/actions/workflows/ci.yml)
![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![PyTest](https://img.shields.io/badge/tests-30%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

A modular diagnostic benchmarking suite designed to probe, quantify, and stress-test spatial reasoning, relative directional comprehension, and topological predicates in Vision-Language Models (VLMs) across progressive photometric degradation tiers.

---

## Architecture Overview

```mermaid
flowchart TD
    Dataset[Synthetic / Real Dataset Ingestion] --> DegradationEngine[Photometric Degradation Engine]
    
    subgraph Degradation Tiers
        DegradationEngine -->|Severity 0| Clean[Original Image]
        DegradationEngine -->|Severity 1-3| Perturbed[Low-Light / Blur / Contrast Noise]
    end

    Clean --> Evaluator[VLMSpatialEvaluator]
    Perturbed --> Evaluator

    subgraph Dual Evaluation Engine
        Evaluator --> CLIPScore[CLIP Cross-Modal Representation Matching]
        Evaluator --> Directional[DirectionalEvaluator: IoU & Topological Predicates]
    end

    Directional --> Predicates[left_of / right_of / above / below / inside / disjoint]
    CLIPScore --> Aggregator[SpatialMetricsAggregator]
    Predicates --> Aggregator

    subgraph Export Engine
        Aggregator --> MD[GitHub-Flavored Markdown Report]
        Aggregator --> LaTeX[Academic LaTeX booktabs Table]
        Aggregator --> JSONDump[Structured Benchmark JSON]
    end