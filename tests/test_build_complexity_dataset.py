import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_complexity_dataset import (  # noqa: E402
    build_dataset,
    phrase_count,
    prompt_texts,
    structured_error,
)


def request(model, items):
    return {"model": model, "input": items, "tools": []}


class ComplexityDatasetTests(unittest.TestCase):
    def test_both_message_schemas_are_extracted(self):
        gpt = request("gpt", [
            {"role": "system", "content": "plain system"},
            {"role": "user", "content": "plain user"},
        ])
        claude = request("claude", [
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "part system"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "part user"}],
            },
        ])
        self.assertEqual(prompt_texts(gpt), ("plain system", ["plain user"]))
        self.assertEqual(prompt_texts(claude), ("part system", ["part user"]))

    def test_phrase_matching_uses_word_boundaries(self):
        self.assertEqual(phrase_count("if this fails", ("if",)), 1)
        self.assertEqual(phrase_count("file diff difficult", ("if",)), 0)
        self.assertEqual(phrase_count("Use an API, not apiculture", ("api",)), 1)

    def test_structured_errors_distinguish_success(self):
        self.assertTrue(structured_error('{"exit_code": 2, "output": "failed"}'))
        self.assertTrue(structured_error({"success": False}))
        self.assertFalse(structured_error('{"error": null, "status": 200}'))

    def test_linked_and_cross_model_rows_share_one_split_group(self):
        opening = [
            {"type": "message", "role": "system", "content": "system"},
            {"type": "message", "role": "user", "content": "build a report"},
        ]
        continued = opening + [
            {"type": "function_call", "name": "search", "arguments": "{}"},
            {"type": "function_call_output", "output": "ok"},
        ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export = root / "export"
            output = root / "results"
            export.mkdir()
            rows = [
                request("model-a", opening),
                request("model-a", continued),
                request("model-b", opening),
            ]
            with (export / "sample.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            manifest = build_dataset(export, output, seed=7)
            generated = [json.loads(line) for line in (output / "all.jsonl").read_text().splitlines()]

            self.assertEqual(manifest["request_count"], 3)
            self.assertEqual(manifest["trajectory_count"], 2)
            self.assertEqual(manifest["split_group_count"], 1)
            self.assertEqual(len({row["split"] for row in generated}), 1)
            self.assertEqual(
                len({row["metadata"]["split_group_id"] for row in generated}), 1
            )
            model_a = [row for row in generated if row["metadata"]["logged_model"] == "model-a"]
            self.assertEqual(len({row["metadata"]["trajectory_id"] for row in model_a}), 1)
            self.assertEqual(sorted(row["metadata"]["request_position"] for row in model_a), [0, 1])

            expected_input = {
                "request_id", "system_prompt", "user_prompt", "user_messages",
                "static_text_features",
            }
            expected_target = {
                "request_id", "complexity_score", "complexity_band",
                "intrinsic_complexity", "observed_difficulty", "observed_weight",
                "label_confidence", "intrinsic_components", "observed_components",
                "observed_metrics",
            }
            expected_metadata = {
                "request_id", "source", "source_line", "split_group_id",
                "trajectory_id", "request_position", "trajectory_size",
                "logged_model", "observed_uses_future_snapshot",
            }
            self.assertEqual(set(generated[0]["input"]), expected_input)
            self.assertEqual(set(generated[0]["target"]), expected_target)
            self.assertEqual(set(generated[0]["metadata"]), expected_metadata)

    def test_normalized_near_duplicates_share_a_split(self):
        rows = [
            request("model-a", [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "Build report 123 for <COMPANY_A> using sales data"},
            ]),
            request("model-b", [
                {"role": "system", "content": "different system"},
                {"role": "user", "content": "Build report 999 for <COMPANY_B> using sales data"},
            ]),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export = root / "export"
            output = root / "results"
            export.mkdir()
            with (export / "sample.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            build_dataset(export, output)
            generated = [json.loads(line) for line in (output / "all.jsonl").read_text().splitlines()]
            self.assertEqual(len({row["split"] for row in generated}), 1)
            self.assertEqual(len({row["metadata"]["split_group_id"] for row in generated}), 1)

    def test_split_assignment_is_reproducible(self):
        rows = []
        for index in range(30):
            items = [
                {"type": "message", "role": "system", "content": "system"},
                {"type": "message", "role": "user", "content": f"task {index}"},
            ]
            rows.append(request("model-a", items))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export = root / "export"
            export.mkdir()
            with (export / "sample.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            first = root / "first"
            second = root / "second"
            build_dataset(export, first, seed=42)
            build_dataset(export, second, seed=42)
            self.assertEqual(
                (first / "all.jsonl").read_bytes(),
                (second / "all.jsonl").read_bytes(),
            )

    def test_test_only_system_outlier_does_not_change_training_caps(self):
        rows = [
            request("model-a", [
                {"role": "system", "content": "system"},
                {"role": "user", "content": f"Create unique_task_{index}"},
            ])
            for index in range(40)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export = root / "export"
            export.mkdir()

            def write_rows():
                with (export / "sample.jsonl").open("w") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")

            write_rows()
            first = root / "first"
            first_manifest = build_dataset(export, first, seed=42)
            metadata = [json.loads(line) for line in (first / "test_metadata.jsonl").read_text().splitlines()]
            test_line = metadata[0]["source_line"] - 1
            rows[test_line]["input"][0]["content"] = "unique system context " * 50_000
            write_rows()
            second_manifest = build_dataset(export, root / "second", seed=42)
            self.assertEqual(
                first_manifest["intrinsic_p95_scaling_caps"],
                second_manifest["intrinsic_p95_scaling_caps"],
            )


if __name__ == "__main__":
    unittest.main()
