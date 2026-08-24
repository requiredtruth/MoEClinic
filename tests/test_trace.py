import json
import math
from pathlib import Path
import tempfile
import unittest

from moeclinic.trace import TraceError, load_jsonl, parse_record


class TraceTests(unittest.TestCase):
    def test_raw_trace_computes_assignments_and_stable_logsumexp(self) -> None:
        trace = parse_record(
            {
                "step": 7,
                "router_probs": [[0.75, 0.25], [0.20, 0.80]],
                "selected_experts": [0, [1]],
                "router_logits": [[0.0, 0.0], [0.0, 0.0]],
            },
            1,
        )
        self.assertEqual(trace.assignment_counts, (1, 1))
        self.assertEqual(trace.probability_sums, (0.95, 1.05))
        self.assertAlmostEqual(trace.logsumexp_sq_sum or 0.0, 2 * math.log(2) ** 2)
        self.assertEqual(trace.logit_token_count, 2)

    def test_compact_trace_loads_from_jsonl(self) -> None:
        record = {
            "step": 1,
            "token_count": 100,
            "assignment_counts": [25, 25, 25, 25],
            "probability_sums": [25, 25, 25, 25],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            traces = load_jsonl(path)
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].experts, 4)

    def test_rejects_probability_mass_that_is_not_one(self) -> None:
        with self.assertRaisesRegex(TraceError, "must sum to 1"):
            parse_record({"router_probs": [[0.2, 0.2]]}, 1)

    def test_rejects_expert_count_change(self) -> None:
        records = [
            {"token_count": 2, "assignment_counts": [1, 1], "probability_sums": [1, 1]},
            {"token_count": 3, "assignment_counts": [1, 1, 1], "probability_sums": [1, 1, 1]},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            path.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
            with self.assertRaisesRegex(TraceError, "expert count changes"):
                load_jsonl(path)


if __name__ == "__main__":
    unittest.main()
