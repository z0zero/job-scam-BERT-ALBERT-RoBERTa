import unittest

from src.models.auth_service import AuthSession, AuthenticatedUser
from src.models.session_store import (
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
    save_auth_session,
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
