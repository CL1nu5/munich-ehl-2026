import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "team"))

from eval.quality import logged_outcome_quality  # noqa: E402


class OutcomeQualityTests(unittest.TestCase):
    def test_quality_is_recovered_tool_success_rate(self):
        target = {"observed_metrics": {"tool_output_count": 8, "tool_error_count": 2}}
        self.assertEqual(logged_outcome_quality(target), 0.75)

    def test_no_recovered_outputs_uses_neutral_fallback(self):
        self.assertEqual(logged_outcome_quality({"observed_metrics": {}}), 0.5)

    def test_error_count_is_clamped_to_valid_quality_range(self):
        target = {"observed_metrics": {"tool_output_count": 1, "tool_error_count": 2}}
        self.assertEqual(logged_outcome_quality(target), 0.0)


if __name__ == "__main__":
    unittest.main()
