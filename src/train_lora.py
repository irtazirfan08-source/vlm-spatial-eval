import argparse
import gc
import json
import sys
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model

sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from transformers import AutoModelForImageTextToText as AutoModelForVLM
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as AutoModelForVLM
    except ImportError:
        from transformers import AutoModel as AutoModelForVLM


class SpatialInstructionDataset(Dataset):
    def __init__(self, metadata_path: str, processor: AutoProcessor, max_samples: int = None):
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        if max_samples:
            self.records = self.records[:max_samples]
        self.processor = processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        item = self.records[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        conversations = item["conversations"]

        full_text = self.processor.apply_chat_template(conversations, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(conversations[:1], add_generation_prompt=True)

        full_inputs = self.processor(text=full_text, images=[image], return_tensors="pt")
        prompt_inputs = self.processor(text=prompt_text, images=[image], return_tensors="pt")

        input_ids = full_inputs["input_ids"].squeeze(0)
        pixel_values = full_inputs.get("pixel_values", None)
        if pixel_values is not None:
            pixel_values = pixel_values.squeeze(0)

        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        data_item = {"input_ids": input_ids, "labels": labels}
        if pixel_values is not None:
            data_item["pixel_values"] = pixel_values
        if "pixel_attention_mask" in full_inputs:
            data_item["pixel_attention_mask"] = full_inputs["pixel_attention_mask"].squeeze(0)

        return data_item


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    padded_inputs = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (padded_inputs != 0).long()

    collated = {
        "input_ids": padded_inputs,
        "labels": padded_labels,
        "attention_mask": attention_mask,
    }

    if "pixel_values" in batch[0]:
        collated["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])
    if "pixel_attention_mask" in batch[0]:
        collated["pixel_attention_mask"] = torch.stack([item["pixel_attention_mask"] for item in batch])

    return collated


def train_lora(
    model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
    metadata_train: str = "data/fine_tune/train/metadata.json",
    output_dir: str = "adapters/smolvlm-spatial-lora",
    epochs: int = 1,
    lr: float = 2e-4,
    batch_size: int = 1,
    grad_accum_steps: int = 2,
    max_samples: int = 20,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Initializing LoRA fine-tuning on device: {device.upper()}")

    processor = AutoProcessor.from_pretrained(model_name)
    base_model = AutoModelForVLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    ).to(device)

    for param in base_model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    train_dataset = SpatialInstructionDataset(metadata_train, processor, max_samples=max_samples)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    total_steps = len(dataloader)
    print(f"--> Beginning fine-tuning ({epochs} epoch, {len(train_dataset)} samples, {total_steps} steps)...\n")

    optimizer.zero_grad()
    for step, batch in enumerate(dataloader, start=1):
        print(f"   [Step {step}/{total_steps}] Computing forward pass...", flush=True)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss / grad_accum_steps
        loss.backward()

        if step % grad_accum_steps == 0 or step == total_steps:
            optimizer.step()
            optimizer.zero_grad()

        print(f"   [Step {step}/{total_steps}] Loss: {loss.item() * grad_accum_steps:.4f}", flush=True)
        gc.collect()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    processor.save_pretrained(str(out_path))
    print(f"\n--> LoRA adapter successfully saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLM using PEFT/LoRA.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    args = parser.parse_args()

    train_lora(
        epochs=args.epochs,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
    )