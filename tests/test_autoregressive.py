"""Unit tests for autoregressive VLM spatial evaluator and token logprob scoring."""

from PIL import Image
import pytest
import torch

from src.autoregressive_evaluator import AutoregressiveSpatialEvaluator


def test_autoregressive_evaluator_mock_logprobs(monkeypatch):
    # Instantiate without downloading weights by mocking init methods
    evaluator = AutoregressiveSpatialEvaluator.__new__(AutoregressiveSpatialEvaluator)
    evaluator.device = "cpu"
    evaluator.yes_token_ids = [101]
    evaluator.no_token_ids = [102]

    class MockInputsDict(dict):
        def to(self, device):
            return self

    class MockProcessor:
        def apply_chat_template(self, messages, add_generation_prompt=True):
            return "mocked prompt text"

        def __call__(self, text, images, return_tensors):
            return MockInputsDict({"input_ids": torch.tensor([[1, 2, 3]])})

    evaluator.processor = MockProcessor()

    # Mock model forward pass returning synthetic logits
    class MockOutputs:
        def __init__(self, logits):
            self.logits = logits

    class MockModel:
        def __call__(self, **kwargs):
            # Shape: (batch_size=1, seq_len=10, vocab_size=200)
            logits = torch.zeros((1, 10, 200))
            logits[0, -1, 101] = 2.5   # higher logit for 'Yes'
            logits[0, -1, 102] = 0.5   # lower logit for 'No'
            return MockOutputs(logits)

    evaluator.model = MockModel()

    dummy_img = Image.new("RGB", (32, 32), color="white")
    res = evaluator.score_sample_logprobs(
        image=dummy_img,
        relation="left_of",
        obj_a=("blue", "square"),
        obj_b=("red", "square")
    )

    assert res["is_correct"] is True
    assert res["yes_logprob"] > res["no_logprob"]
    assert res["margin"] > 0