import json
import tempfile
import unittest
from pathlib import Path

from uocr_train.synth_smoke_data import generate_dataset


class SynthSmokeDataTest(unittest.TestCase):
    def test_generates_single_and_multi_page_markdown_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            train_jsonl = generate_dataset(
                output_dir=out_dir,
                num_single=3,
                num_multi=2,
                pages_per_multi=2,
                seed=123,
            )

            self.assertEqual(train_jsonl, out_dir / "train.jsonl")
            rows = [json.loads(line) for line in train_jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 5)

            single_rows = [row for row in rows if row["mode"] == "single_gundam"]
            multi_rows = [row for row in rows if row["mode"] == "multi_base"]
            self.assertEqual(len(single_rows), 3)
            self.assertEqual(len(multi_rows), 2)

            for row in rows:
                self.assertEqual(row["target_type"], "page_markdown")
                self.assertTrue(row["prompt"].startswith("<image>"))
                self.assertTrue(row["target"].strip())
                self.assertIn("markdown", row["source"])
                for image_path in row["images"]:
                    self.assertTrue((out_dir / image_path).is_file(), image_path)

            for row in single_rows:
                self.assertEqual(len(row["images"]), 1)
                self.assertIn("# ", row["target"])
                self.assertIn("<table>", row["target"])

            for row in multi_rows:
                self.assertEqual(len(row["images"]), 2)
                self.assertEqual(row["target"].count("<PAGE>"), 2)
                self.assertIn("Multi page parsing", row["prompt"])


if __name__ == "__main__":
    unittest.main()
