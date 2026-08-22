import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "team"))

from model.train import load_evaluation_ids  # noqa: E402


class RouterSplitSafetyTests(unittest.TestCase):
    def test_validation_and_test_ids_are_both_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, request_id in (("validation", "v:1"), ("test", "t:1")):
                (root / f"{split}_inputs.jsonl").write_text(
                    json.dumps({"request_id": request_id}) + "\n"
                )

            self.assertEqual(load_evaluation_ids(root), {"v:1", "t:1"})


if __name__ == "__main__":
    unittest.main()
