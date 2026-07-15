import unittest
from types import SimpleNamespace

from src.models.browser_session_store import (
    BrowserSessionSnapshot,
    BrowserSessionStore,
)


class FakeRenderer:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class BrowserSessionStoreTests(unittest.TestCase):
    def test_reads_valid_tokens_returned_by_browser(self):
        renderer = FakeRenderer(
            SimpleNamespace(
                snapshot={
                    "fingerprint": "read",
                    "tokens": {
                        "access_token": "access-a",
                        "refresh_token": "refresh-a",
                    },
                },
            )
        )
        store = BrowserSessionStore(renderer=renderer)

        result = store.sync(None)

        self.assertEqual(
            result,
            BrowserSessionSnapshot(
                ready=True,
                tokens=("access-a", "refresh-a"),
            ),
        )

    def test_rejects_incomplete_tokens_returned_by_browser(self):
        renderer = FakeRenderer(
            SimpleNamespace(
                snapshot={
                    "fingerprint": "read",
                    "tokens": {
                        "access_token": "access-a",
                        "refresh_token": None,
                    },
                },
            )
        )
        store = BrowserSessionStore(renderer=renderer)

        result = store.sync(None)

        self.assertEqual(result, BrowserSessionSnapshot(True, None))

    def test_clear_command_never_returns_stale_tokens(self):
        renderer = FakeRenderer(
            SimpleNamespace(
                snapshot={
                    "fingerprint": "clear",
                    "tokens": {
                        "access_token": "stale-access",
                        "refresh_token": "stale-refresh",
                    },
                },
            )
        )
        store = BrowserSessionStore(renderer=renderer)

        result = store.sync(None, clear=True)

        self.assertEqual(result, BrowserSessionSnapshot(True, None))
        self.assertTrue(renderer.calls[0]["data"]["clear"])


if __name__ == "__main__":
    unittest.main()
