import unittest

from src.controllers.email_verification_notice import consume_email_verification_notice


class EmailVerificationNoticeTests(unittest.TestCase):
    def test_true_marker_is_consumed_and_returns_true(self):
        params = {"verified": "true", "source": "email"}

        self.assertTrue(consume_email_verification_notice(params))
        self.assertEqual(params, {"source": "email"})

    def test_unsupported_marker_is_consumed_and_returns_false(self):
        params = {"verified": "false", "source": "email"}

        self.assertFalse(consume_email_verification_notice(params))
        self.assertEqual(params, {"source": "email"})

    def test_missing_marker_preserves_parameters(self):
        params = {"token_hash": "secret", "type": "recovery"}

        self.assertFalse(consume_email_verification_notice(params))
        self.assertEqual(params, {"token_hash": "secret", "type": "recovery"})


if __name__ == "__main__":
    unittest.main()
