import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_dashboard_data import call_input_ends  # noqa: E402


class RecoveredCallBoundaryTests(unittest.TestCase):
    def test_prior_call_starts_before_model_output_and_current_call_uses_full_input(self):
        items = [
            {"type": "message", "role": "system", "content": "system"},
            {"type": "message", "role": "user", "content": "task"},
            {"type": "reasoning", "summary": []},
            {"type": "function_call", "name": "tool", "arguments": "{}"},
            {"type": "function_call_output", "output": '{"success": true}'},
        ]

        self.assertEqual(call_input_ends(items), [2, 5])

    def test_parallel_tool_calls_are_one_model_output_group(self):
        items = [
            {"type": "message", "role": "user", "content": "task"},
            {"type": "function_call", "name": "one", "arguments": "{}"},
            {"type": "function_call", "name": "two", "arguments": "{}"},
            {"type": "function_call_output", "output": "ok"},
            {"type": "function_call_output", "output": "ok"},
        ]

        self.assertEqual(call_input_ends(items), [1, 5])


if __name__ == "__main__":
    unittest.main()
