import json
from pathlib import Path
from src.train_data_generator import SpatialInstructionDatasetBuilder


def test_spatial_instruction_dataset_builder(tmp_path):
    test_out = tmp_path / "test_data"
    builder = SpatialInstructionDatasetBuilder(output_dir=str(test_out), canvas_size=64)

    # 1 sample per relation = 4 samples * 2 queries (Yes/No) = 8 records
    records = builder.build_split("train", num_samples_per_relation=1)

    assert len(records) == 8
    yes_count = sum(1 for r in records if r["ground_truth"] == "Yes")
    no_count = sum(1 for r in records if r["ground_truth"] == "No")

    # Verify strict 50/50 balance
    assert yes_count == 4
    assert no_count == 4

    # Verify metadata.json was properly written to disk
    metadata_file = test_out / "train" / "metadata.json"
    assert metadata_file.exists()

    with open(metadata_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 8