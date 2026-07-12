import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from src.models.auth_service import (
    AuthError,
    AuthService,
    AuthSession,
    AuthenticatedUser,
    ValidationError,
    normalize_email,
    validate_new_password,
    validate_signup,
)


def auth_response():
    user = SimpleNamespace(
        id="user-a",
        email="person@example.com",
        user_metadata={"full_name": "Person A"},
    )
    session = SimpleNamespace(
        access_token="access-a",
        refresh_token="refresh-a",
        user=user,
    )
    return SimpleNamespace(session=session, user=user)


class AuthValidationTests(unittest.TestCase):
    def test_normalize_email(self):
        self.assertEqual(normalize_email(" Person@Example.COM "), "person@example.com")

    def test_signup_rejects_short_password(self):
        with self.assertRaises(ValidationError):
            validate_signup("Person", "person@example.com", "short", "short")

    def test_new_password_requires_matching_confirmation(self):
        with self.assertRaises(ValidationError):
            validate_new_password("long-password", "different-password")


class AuthServiceTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.service = AuthService(self.client, "https://app.example.com")

    def test_sign_up_sends_full_name_and_redirect(self):
        self.client.auth.sign_up.return_value = SimpleNamespace(session=None)

        self.service.sign_up(
            " Person A ",
            " Person@Example.COM ",
            "password-123",
            "password-123",
        )

        self.client.auth.sign_up.assert_called_once_with(
            {
                "email": "person@example.com",
                "password": "password-123",
                "options": {
                    "data": {"full_name": "Person A"},
                    "email_redirect_to": "https://app.example.com",
                },
            }
        )

    def test_sign_in_returns_stable_session_dataclass(self):
        self.client.auth.sign_in_with_password.return_value = auth_response()

        result = self.service.sign_in("person@example.com", "password-123")

        self.assertEqual(
            result,
            AuthSession(
                user=AuthenticatedUser(
                    id="user-a",
                    email="person@example.com",
                    full_name="Person A",
                ),
                access_token="access-a",
                refresh_token="refresh-a",
            ),
        )

    def test_restore_session_uses_set_session_for_refresh(self):
        self.client.auth.set_session.return_value = auth_response()

        result = self.service.restore_session("old-access", "old-refresh")

        self.client.auth.set_session.assert_called_once_with(
            "old-access", "old-refresh"
        )
        self.assertEqual(result.refresh_token, "refresh-a")

    def test_verify_rejects_unknown_callback_type(self):
        with self.assertRaises(ValidationError):
            self.service.verify_token("token-hash", "invite")

    def test_login_exception_becomes_generic_error(self):
        self.client.auth.sign_in_with_password.side_effect = RuntimeError(
            "user does not exist"
        )

        with self.assertRaisesRegex(AuthError, "Email or password is invalid"):
            self.service.sign_in("person@example.com", "password-123")


if __name__ == "__main__":
    unittest.main()
