import unittest

from src.models.auth_service import AuthSession, AuthenticatedUser
from src.models.session_store import (
    AUTH_NOTICE_KEY,
    AUTH_STATE_KEY,
    MODEL_LOADING_PENDING_KEY,
    RECOVERY_MODE_KEY,
    clear_auth_state,
    is_model_loading_pending,
    is_recovery_mode,
    load_auth_session,
    load_auth_tokens,
    mark_model_loading_pending,
    mark_recovery_mode,
    pop_auth_notice,
    save_auth_session,
    set_auth_notice,
)


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.state = {}
        self.session = AuthSession(
            user=AuthenticatedUser(
                id="user-a",
                email="person@example.com",
                full_name="Person A",
            ),
            access_token="access-a",
            refresh_token="refresh-a",
        )

    def test_save_and_load_round_trip(self):
        save_auth_session(self.state, self.session)
        self.assertEqual(set(self.state), {AUTH_STATE_KEY})
        self.assertEqual(load_auth_session(self.state), self.session)
        self.assertEqual(
            load_auth_tokens(self.state),
            ("access-a", "refresh-a"),
        )

    def test_recovery_marker_is_explicit(self):
        self.assertFalse(is_recovery_mode(self.state))
        mark_recovery_mode(self.state, True)
        self.assertTrue(is_recovery_mode(self.state))

    def test_recovery_marker_rejects_non_true_values(self):
        for value in (False, 1, "true", [True]):
            with self.subTest(value=value):
                mark_recovery_mode(self.state, value)
                self.assertFalse(is_recovery_mode(self.state))

    def test_model_loading_marker_is_explicit_and_removable(self):
        self.assertFalse(is_model_loading_pending(self.state))

        mark_model_loading_pending(self.state, True)
        self.assertTrue(is_model_loading_pending(self.state))
        self.assertEqual(
            self.state[MODEL_LOADING_PENDING_KEY],
            True,
        )

        mark_model_loading_pending(self.state, False)
        self.assertFalse(is_model_loading_pending(self.state))
        self.assertNotIn(MODEL_LOADING_PENDING_KEY, self.state)

    def test_model_loading_marker_rejects_non_true_values(self):
        for value in (False, 1, "true", [True]):
            with self.subTest(value=value):
                self.state[MODEL_LOADING_PENDING_KEY] = value
                self.assertFalse(
                    is_model_loading_pending(self.state)
                )

    def test_auth_notice_round_trip_is_consumed_once(self):
        set_auth_notice(
            self.state,
            "Password updated. Sign in with your new password.",
        )

        self.assertEqual(
            pop_auth_notice(self.state),
            "Password updated. Sign in with your new password.",
        )
        self.assertIsNone(pop_auth_notice(self.state))
        self.assertNotIn(AUTH_NOTICE_KEY, self.state)

    def test_empty_auth_notice_is_not_stored(self):
        set_auth_notice(self.state, "")

        self.assertNotIn(AUTH_NOTICE_KEY, self.state)
        self.assertIsNone(pop_auth_notice(self.state))

    def test_invalid_auth_notice_value_is_discarded(self):
        for value in (None, 1, True, [], {}):
            with self.subTest(value=value):
                self.state[AUTH_NOTICE_KEY] = value
                self.assertIsNone(pop_auth_notice(self.state))
                self.assertNotIn(AUTH_NOTICE_KEY, self.state)

    def test_clear_auth_state_preserves_pending_notice(self):
        save_auth_session(self.state, self.session)
        mark_recovery_mode(self.state, True)
        mark_model_loading_pending(self.state, True)
        set_auth_notice(
            self.state,
            "Password updated. Sign in with your new password.",
        )

        clear_auth_state(self.state)

        self.assertIsNone(load_auth_session(self.state))
        self.assertFalse(is_recovery_mode(self.state))
        self.assertFalse(is_model_loading_pending(self.state))
        self.assertEqual(
            pop_auth_notice(self.state),
            "Password updated. Sign in with your new password.",
        )

    def test_clear_removes_tokens_identity_recovery_and_loading_state(self):
        save_auth_session(self.state, self.session)
        mark_recovery_mode(self.state, True)
        mark_model_loading_pending(self.state, True)

        clear_auth_state(self.state)

        self.assertNotIn(AUTH_STATE_KEY, self.state)
        self.assertNotIn(RECOVERY_MODE_KEY, self.state)
        self.assertNotIn(MODEL_LOADING_PENDING_KEY, self.state)
        self.assertIsNone(load_auth_session(self.state))
        self.assertIsNone(load_auth_tokens(self.state))
        self.assertFalse(is_recovery_mode(self.state))
        self.assertFalse(is_model_loading_pending(self.state))

    def test_malformed_partial_state_is_rejected(self):
        self.state[AUTH_STATE_KEY] = {
            "user_id": "user-a",
            "access_token": "access-a",
        }
        self.assertIsNone(load_auth_session(self.state))
        self.assertIsNone(load_auth_tokens(self.state))

    def test_non_mapping_auth_state_is_rejected(self):
        for value in (None, "auth", ["auth"]):
            with self.subTest(value=value):
                self.state[AUTH_STATE_KEY] = value
                self.assertIsNone(load_auth_session(self.state))
                self.assertIsNone(load_auth_tokens(self.state))

    def test_non_string_auth_fields_are_rejected(self):
        save_auth_session(self.state, self.session)
        valid_payload = self.state[AUTH_STATE_KEY]
        for field in valid_payload:
            with self.subTest(field=field):
                self.state[AUTH_STATE_KEY] = {
                    **valid_payload,
                    field: 1,
                }
                self.assertIsNone(load_auth_session(self.state))
                self.assertIsNone(load_auth_tokens(self.state))


if __name__ == "__main__":
    unittest.main()
