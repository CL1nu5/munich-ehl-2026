import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from load_trajectories import group_trajectories  # noqa: E402


def request(user_text, model, extra_items=None):
    items = [
        {"type": "message", "role": "system", "content": "system"},
        {"type": "message", "role": "user", "content": user_text},
    ]
    items.extend(extra_items or [])
    return {"model": model, "input": items, "tools": []}


class TrajectoryGroupingTests(unittest.TestCase):
    def test_exact_prefix_requests_form_one_trajectory(self):
        first = request("same task", "model-a")
        second = request(
            "same task",
            "model-a",
            [{"type": "function_call", "name": "tool", "arguments": "{}"}],
        )

        groups = group_trajectories([second, first])

        self.assertEqual(len(groups), 1)
        self.assertEqual(next(iter(groups.values())), [first, second])

    def test_reused_prompt_without_prefix_is_not_merged(self):
        left = request("boilerplate", "model-a", [{"type": "message", "role": "assistant", "content": "left"}])
        right = request("boilerplate", "model-b", [{"type": "message", "role": "assistant", "content": "right"}])

        groups = group_trajectories([left, right])

        self.assertEqual(len(groups), 2)

    def test_complete_user_text_is_used_instead_of_2000_char_prefix(self):
        prefix = "x" * 2000
        groups = group_trajectories([
            request(prefix + "a", "model-a"),
            request(prefix + "b", "model-a"),
        ])
        self.assertEqual(len(groups), 2)


if __name__ == "__main__":
    unittest.main()
