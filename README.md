# VLM-SpatialEval: Spatial Reasoning & Robustness Benchmark for Vision-Language Models

An automated diagnostic evaluation harness designed to stress-test Vision-Language Models (VLMs) like CLIP, SigLIP, and PaliGemma on spatial-relational reasoning under controlled photometric degradation.

## Motivation
Standard VLM benchmarks predominantly evaluate models under pristine lighting and compositional framing. However, cross-modal alignment degrades non-linearly when visual inputs suffer from physical degradations such as underexposure and sensor noise. Furthermore, models frequently rely on "bag-of-words" heuristics rather than resolving directional spatial syntax.

`VLM-SpatialEval` systematically probes:
1. **Photometric Collapse Thresholds:** Quantifying the exact lux-attenuation curve where visual token alignment breaks down.
2. **Spatial-Relational Inversion:** Testing model resilience against directional counterfactuals ("above" vs. "below", "to the left of" vs. "to the right of").
3. **Internal Representation Probing:** Evaluating whether intermediate transformer layers preserve spatial topology prior to multimodal projection.

## Repository Structure