from PIL import Image
from src.evaluator import VLMSpatialEvaluator


def test_evaluator_score_pair_structure():
    evaluator = VLMSpatialEvaluator(model_name="openai/clip-vit-base-patch32", device="cpu")
    img = Image.new("RGB", (64, 64), color="red")
    
    result = evaluator.score_pair(
        image=img,
        positive_text="a red square",
        negative_text="a blue triangle",
    )

    assert "sim_positive" in result
    assert "sim_negative" in result
    assert "is_correct" in result
    assert "margin" in result
    assert isinstance(result["is_correct"], bool)
    assert isinstance(result["margin"], float)
    assert -1.0 <= result["sim_positive"] <= 1.0
    assert -1.0 <= result["sim_negative"] <= 1.0


def test_evaluator_dataset_execution():
    evaluator = VLMSpatialEvaluator(model_name="openai/clip-vit-base-patch32", device="cpu")
    dummy_sample = {
        "image": Image.new("RGB", (64, 64), color="blue"),
        "positive_prompt": "a blue square",
        "negative_prompt": "a red circle",
    }
    
    metrics = evaluator.evaluate_dataset_at_severity([dummy_sample], severity=0)

    assert metrics["severity_level"] == 0
    assert metrics["total_samples"] == 1
    assert "accuracy" in metrics
    assert "mean_margin" in metrics