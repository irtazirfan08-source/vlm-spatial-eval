from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor
from src.train_data_generator import SpatialInstructionDatasetBuilder
from src.train_lora import AutoModelForVLM, SpatialInstructionDataset, collate_fn


def test_lora_dataset_and_forward_pass(tmp_path):
    # Automatically generate a tiny temporary split inside CI sandbox
    builder = SpatialInstructionDatasetBuilder(output_dir=str(tmp_path), canvas_size=64)
    builder.build_split("val", num_samples_per_relation=1)

    metadata_path = tmp_path / "val" / "metadata.json"
    assert metadata_path.exists(), "Temporary validation metadata was not created."

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    dataset = SpatialInstructionDataset(str(metadata_path), processor, max_samples=2)

    assert len(dataset) == 2
    item = dataset[0]
    assert "input_ids" in item
    assert "labels" in item
    assert -100 in item["labels"], "Prompt tokens must be masked with -100."

    # Verify batch collation
    batch = collate_fn([dataset[0], dataset[1]])
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape[0] == 2

    # Verify LoRA initialization and forward pass
    base_model = AutoModelForVLM.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct", torch_dtype=torch.float32
    )
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)

    with torch.no_grad():
        outputs = model(**batch)
        assert outputs.loss is not None
        assert not torch.isnan(outputs.loss)