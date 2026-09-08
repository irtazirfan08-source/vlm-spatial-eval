from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoProcessor
from src.evaluate_lora import AutoModelForVLM


def test_lora_adapter_loading():
    adapter_path = Path("adapters/smolvlm-spatial-lora")
    assert adapter_path.exists(), "Trained LoRA adapter directory does not exist."
    assert (adapter_path / "adapter_model.safetensors").exists() or (adapter_path / "adapter_model.bin").exists()

    processor = AutoProcessor.from_pretrained(str(adapter_path))
    assert processor is not None

    base_model = AutoModelForVLM.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct", torch_dtype=torch.float32
    )
    lora_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    assert isinstance(lora_model, PeftModel)