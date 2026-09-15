import json
import tempfile
import unittest
from pathlib import Path

from format_for_hf import convert_to_diffusers_format
from sample_captures import sample_images


class DatasetToolTests(unittest.TestCase):
    def test_sampling_is_seeded_and_non_destructive_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for index in range(5):
                (source / f"{index}.jpg").write_bytes(b"fixture")

            first = sample_images(source, root / "first", 3, seed=7)
            second = sample_images(source, root / "second", 3, seed=7)
            self.assertEqual([path.name for path in first], [path.name for path in second])
            with self.assertRaises(FileExistsError):
                sample_images(source, root / "first", 1)

    def test_diffusers_conversion_pairs_images_and_captions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            concept = root / "source" / "concept_a"
            concept.mkdir(parents=True)
            (concept / "example.jpg").write_bytes(b"fixture")
            (concept / "example.txt").write_text("sks_example, a test image", encoding="utf-8")

            output = root / "output"
            metadata = convert_to_diffusers_format(root / "source", output)

            self.assertEqual(len(metadata), 1)
            self.assertTrue((output / "concept_a_example.jpg").exists())
            line = json.loads((output / "metadata.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(line["text"], "sks_example, a test image")


if __name__ == "__main__":
    unittest.main()
