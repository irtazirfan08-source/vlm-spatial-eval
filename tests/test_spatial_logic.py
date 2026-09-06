from PIL import Image
from src.spatial_generator import SpatialDatasetGenerator, RELATION_INVERSES


def test_directional_sample_structure():
    generator = SpatialDatasetGenerator(canvas_size=256)
    sample = generator.generate_directional_sample("to the left of")

    assert isinstance(sample["image"], Image.Image)
    assert sample["image"].size == (256, 256)
    assert sample["target_relation"] == "to the left of"
    assert "to the left of" in sample["positive_prompt"]
    assert "to the right of" in sample["negative_prompt"]
    assert sample["positive_prompt"] != sample["negative_prompt"]


def test_benchmark_split_balance():
    generator = SpatialDatasetGenerator(canvas_size=128)
    split = generator.generate_benchmark_split(num_samples_per_relation=5)

    expected_total = 5 * len(RELATION_INVERSES)
    assert len(split) == expected_total

    # Verify each relation occurs in equal proportions
    counts = {rel: 0 for rel in RELATION_INVERSES.keys()}
    for item in split:
        counts[item["target_relation"]] += 1

    for rel, count in counts.items():
        assert count == 5, f"Relation {rel} expected 5 instances, got {count}"