## Empirical Findings (CLIP ViT-B/32)

| Severity Level | Attenuation / Noise | Accuracy (%) | Mean Margin |
| :--- | :--- | :--- | :--- |
| **$L_0$** | Clean Baseline | 28.1% | -0.0024 |
| **$L_1$** | Mild Attenuation ($\gamma=1.5, \sigma=5$) | 25.0% | -0.0020 |
| **$L_2$** | Moderate Starvation ($\gamma=2.2, \sigma=15$) | 37.5% | -0.0013 |
| **$L_3$** | Severe Starvation ($\gamma=3.0, \sigma=30$) | 37.5% | -0.0012 |
| **$L_4$** | Extreme Low-Light ($\gamma=4.2, \sigma=60$) | 53.1% | 0.0000 |

### Key Observations
* **Syntactic Blindness:** At baseline ($L_0$), the model achieves only **28.1% accuracy** (substantially below the 50% random chance baseline for binary counterfactuals). This confirms the *bag-of-words* hypothesis: the model attends to object tokens ("square", "circle") but fails to encode directional syntax ("to the left of" vs. "to the right of").
* **Degradation Convergence:** As photometric degradation reaches $L_4$, the cosine similarity margin shrinks to $0.0000$, indicating visual token collapse where the model degenerates into pure random guessing.

![Degradation Curve](perception_collapse_curve.png)