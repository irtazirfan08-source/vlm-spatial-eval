## Post-Training Intervention: Parameter-Efficient Fine-Tuning (PEFT / LoRA)

To address the spatial reasoning failure identified during benchmarking, we implemented a supervised instruction alignment pipeline targeting SmolVLM-256M:

1. **Synthetic Conversational Dataset:** Generated balanced relational query pairs with hard negative counterfactuals across canonical spatial prepositions.
2. **Adapter Architecture:** Injected low-rank adapters ($r=8, \alpha=16$) into the language decoder's query and value projections (`q_proj`, `v_proj`), representing **0.29% of total parameters** (755,712 trainable weights).
3. **Target Loss Masking:** Formatted conversations using SmolVLM chat templates with prompt tokens masked to `-100` to focus cross-entropy loss exclusively on target outputs.

### Results on Held-Out Validation Split

| Model Variant | Relational Accuracy (%) | Trainable Params (%) | General Language Retention |
| :--- | :--- | :--- | :--- |
| **SmolVLM-256M (Baseline)** | 66.7% | 0.0% | Normal |
| **SmolVLM-256M + Spatial-LoRA** | **75.0%** | **0.29%** | **Preserved ("White")** |

**Intervention Delta:** **+8.3% absolute accuracy gain** on balanced spatial discrimination without language degradation.