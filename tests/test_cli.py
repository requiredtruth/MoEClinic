from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from moeclinic.cli import main


class CliTests(unittest.TestCase):
    def _trace(self, directory: str, counts: list[int]) -> Path:
        total = sum(counts)
        path = Path(directory) / "trace.jsonl"
        path.write_text(
            json.dumps(
                {
                    "token_count": total,
                    "assignment_counts": counts,
                    "probability_sums": [float(value) for value in counts],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def test_json_report_and_collapse_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._trace(temporary, [90, 10, 0, 0])
            output = StringIO()
            with redirect_stdout(output):
                status = main([str(path), "--json"])
        payload = json.loads(output.getvalue())
        self.assertEqual(status, 1)
        self.assertEqual(payload["severity"], "collapse")

    def test_never_fail_mode_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._trace(temporary, [90, 10, 0, 0])
            with redirect_stdout(StringIO()):
                status = main([str(path), "--fail-on", "never"])
        self.assertEqual(status, 0)

    def test_invalid_trace_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.jsonl"
            path.write_text("not json\n", encoding="utf-8")
            with redirect_stderr(StringIO()):
                status = main([str(path)])
        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
