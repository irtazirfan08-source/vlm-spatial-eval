import argparse
import json
import sys
from pathlib import Path
from PIL import Image
import torch
from peft import PeftModel
from transformers import AutoProcessor

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from transformers import AutoModelForImageTextToText as AutoModelForVLM
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as AutoModelForVLM
    except ImportError:
        from transformers import AutoModel as AutoModelForVLM


def evaluate_model(model, processor, dataset, device: str) -> float:
    """Evaluates binary accuracy on balanced spatial query pairs."""
    correct = 0
    total = len(dataset)

    for idx, item in enumerate(dataset, start=1):
        image = Image.open(item["image_path"]).convert("RGB")
        prompt_text = processor.apply_chat_template(
            item["conversations"][:1], add_generation_prompt=True
        )
        inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)

        generated_text = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip().lower()

        ground_truth = item["ground_truth"].strip().lower()
        is_correct = (ground_truth in generated_text) and (
            "no" not in generated_text if ground_truth == "yes" else "yes" not in generated_text
        )

        if is_correct:
            correct += 1

        print(f"      Sample {idx}/{total} -> Pred: '{generated_text}' | GT: '{ground_truth}' | {'✓' if is_correct else '✗'}", flush=True)

    accuracy = (correct / total) * 100.0 if total else 0.0
    return round(accuracy, 2)


def test_language_retention(model, processor, device: str):
    """Verifies that the adapter does not cause catastrophic forgetting on general queries."""
    print("\n--> Testing General Language Retention (Anti-Degeneration Check)...")
    dummy_image = Image.new("RGB", (224, 224), color=(255, 255, 255))
    prompt = "Question: What color is this image? Answer with one word."
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[dummy_image], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    response = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0].strip()
    print(f"    Retention Prompt: '{prompt}'")
    print(f"    Model Response:   '{response}'\n")


def run_comparative_eval(
    adapter_path: str = "adapters/smolvlm-spatial-lora",
    base_model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
    val_metadata: str = "data/fine_tune/val/metadata.json",
    eval_samples: int = 12,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 65)
    print("   SmolVLM Spatial Alignment: Baseline vs. LoRA Comparison")
    print("=" * 65 + "\n")

    processor = AutoProcessor.from_pretrained(base_model_name)
    with open(val_metadata, "r", encoding="utf-8") as f:
        val_data = json.load(f)[:eval_samples]

    # 1. Baseline Evaluation (Un-adapted base model)
    print(f"--> [1/2] Evaluating Zero-Shot Base Model ({len(val_data)} validation samples)...")
    base_model = AutoModelForVLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    ).to(device).eval()

    base_accuracy = evaluate_model(base_model, processor, val_data, device)
    print(f"    Base Model Accuracy: {base_accuracy}%\n")

    # 2. LoRA-Adapted Evaluation
    print(f"--> [2/2] Loading LoRA Adapter from '{adapter_path}' and Evaluating...")
    lora_model = PeftModel.from_pretrained(base_model, adapter_path).to(device).eval()
    lora_accuracy = evaluate_model(lora_model, processor, val_data, device)
    print(f"    LoRA-Adapted Accuracy: {lora_accuracy}%\n")

    # 3. Retention Check
    test_language_retention(lora_model, processor, device)

    # 4. Summary Table
    delta = lora_accuracy - base_accuracy
    print("=" * 65)
    print(f"| Model Variant                | Relational Accuracy (%) |")
    print(f"|:-----------------------------|:------------------------|")
    print(f"| SmolVLM-256M (Baseline)      | {base_accuracy:<23.1f} |")
    print(f"| SmolVLM-256M + Spatial-LoRA  | {lora_accuracy:<23.1f} |")
    print("=" * 65)
    print(f"Absolute Intervention Delta: {'+' if delta >= 0 else ''}{delta:.1f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SmolVLM baseline vs. LoRA adapter.")
    parser.add_argument("--eval_samples", type=int, default=12, help="Number of held-out validation queries to test.")
    args = parser.parse_args()

    run_comparative_eval(eval_samples=args.eval_samples)