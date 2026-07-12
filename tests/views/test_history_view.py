import unittest

from src.views.history_view import format_confidence, make_snippet


class HistoryFormattingTests(unittest.TestCase):
    def test_make_snippet_collapses_whitespace_and_truncates(self):
        text = "  first\nsecond  " + ("x" * 250)
        result = make_snippet(text, limit=20)
        self.assertEqual(result, "first second xxxxxxx…")

    def test_format_confidence(self):
        self.assertEqual(format_confidence(0.9234), "92.3%")


if __name__ == "__main__":
    unittest.main()
