# VLM Spatial Reasoning & Robustness Evaluation Engine (`vlm-spatial-eval`)

[![CI Pipeline](https://github.com/irtazirfan08-source/vlm-spatial-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/irtazirfan08-source/vlm-spatial-eval/actions/workflows/ci.yml)
![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![PyTest](https://img.shields.io/badge/tests-34%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

A modular diagnostic benchmarking suite designed to probe, quantify, and stress-test spatial reasoning, relative directional comprehension, and topological predicates in Vision-Language Models (VLMs) across progressive photometric degradation tiers.

---

## Architecture Overview

```mermaid
flowchart TD
    Dataset[Synthetic / Real Dataset Ingestion] --> DegradationEngine[Photometric Degradation Engine]
    
    subgraph Degradation Tiers
        DegradationEngine -->|Severity L0| Clean[Original Image]
        DegradationEngine -->|Severity L1-L4| Perturbed[Low-Light / Blur / Contrast Noise]
    end

    Clean --> EvaluatorRouter{Evaluator Architecture}
    Perturbed --> EvaluatorRouter

    subgraph Evaluation Engine
        EvaluatorRouter -->|--evaluator-type clip / siglip| Contrastive[VLMSpatialEvaluator: Cross-Modal Matching]
        EvaluatorRouter -->|--evaluator-type autoregressive| Autoregressive[AutoregressiveSpatialEvaluator: Token LogProbs]
        EvaluatorRouter -->|Directional| Logic[DirectionalEvaluator: BBox IoU & Topology]
    end

    Logic --> Predicates[left_of / right_of / above / below / inside / disjoint]
    Contrastive --> Aggregator[SpatialMetricsAggregator]
    Autoregressive --> Aggregator
    Predicates --> Aggregator

    subgraph Intervention Pipeline
        Autoregressive --> LoRA[adapters/smolvlm-spatial-lora]
        LoRA --> PostEval[Post-Intervention Delta Check]
    end

    subgraph Export & Diagnostic Engine
        Aggregator --> MD[GitHub-Flavored Markdown Report]
        Aggregator --> LaTeX[Academic LaTeX booktabs Table]
        Aggregator --> JSONDump[Structured Benchmark JSON]
        Aggregator --> Visualizer[Matplotlib Robustness Curves & Failure Overlays]
    end