from PIL import Image
from src.autoregressive_evaluator import AutoregressiveSpatialEvaluator


def test_autoregressive_evaluator_interface():
    evaluator = AutoregressiveSpatialEvaluator(
        model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
        device="cpu"
    )
    img = Image.new("RGB", (64, 64), color="white")
    
    # Verify execution pipeline returns a boolean verdict
    verdict = evaluator.evaluate_sample(
        image=img,
        relation="above",
        obj_a=("red", "circle"),
        obj_b=("blue", "square"),
    )
    assert isinstance(verdict, bool)