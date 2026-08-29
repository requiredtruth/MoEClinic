import unittest

from moeclinic.gui_support import demo_arguments


class GuiSupportTests(unittest.TestCase):
    def test_empty_arguments_select_safe_bundled_demo(self) -> None:
        self.assertEqual(
            demo_arguments(""),
            ("examples/healthy.jsonl", "--fail-on", "never"),
        )

    def test_operator_arguments_are_preserved(self) -> None:
        self.assertEqual(
            demo_arguments('"my trace.jsonl" --json'),
            ("my trace.jsonl", "--json"),
        )

    def test_malformed_arguments_fail_instead_of_being_reinterpreted(self) -> None:
        with self.assertRaises(ValueError):
            demo_arguments('"unterminated')


if __name__ == "__main__":
    unittest.main()
