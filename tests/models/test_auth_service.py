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

    def test_sign_up_marks_default_confirmation_redirect(self):
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
                    "email_redirect_to": "https://app.example.com/?verified=true",
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

    def test_restore_session_rejects_empty_or_whitespace_access_token(self):
        for access_token in ("", "   "):
            with self.subTest(access_token=access_token):
                self.client.auth.set_session.reset_mock()

                with self.assertRaises(ValidationError):
                    self.service.restore_session(access_token, "refresh-token")

                self.client.auth.set_session.assert_not_called()

    def test_restore_session_rejects_empty_or_whitespace_refresh_token(self):
        for refresh_token in ("", "   "):
            with self.subTest(refresh_token=refresh_token):
                self.client.auth.set_session.reset_mock()

                with self.assertRaises(ValidationError):
                    self.service.restore_session("access-token", refresh_token)

                self.client.auth.set_session.assert_not_called()

    def test_recovery_rejects_whitespace_token_without_calling_provider(self):
        with self.assertRaises(ValidationError):
            self.service.verify_recovery_token("   ")

        self.client.auth.verify_otp.assert_not_called()

    def test_recovery_uses_exact_signature_without_altering_opaque_token(self):
        self.client.auth.verify_otp.return_value = auth_response()

        result = self.service.verify_recovery_token(" opaque token ")

        self.client.auth.verify_otp.assert_called_once_with(
            {"token_hash": " opaque token ", "type": "recovery"}
        )
        self.assertEqual(result.access_token, "access-a")

    def test_reset_request_uses_exact_signature(self):
        self.service.request_password_reset(" Person@Example.COM ")

        self.client.auth.reset_password_for_email.assert_called_once_with(
            "person@example.com",
            {"redirect_to": "https://app.example.com"},
        )

    def test_reset_request_provider_failure_is_generic_and_chained(self):
        provider_error = RuntimeError("registered email leaked")
        self.client.auth.reset_password_for_email.side_effect = provider_error

        with self.assertRaises(AuthError) as raised:
            self.service.request_password_reset("person@example.com")

        self.assertEqual(
            str(raised.exception),
            "If an account exists for that email, a reset message will be sent.",
        )
        self.assertNotIn("registered email leaked", str(raised.exception))
        self.assertIs(raised.exception.__cause__, provider_error)

    def test_update_password_uses_exact_signature(self):
        self.service.update_password("password-123", "password-123")

        self.client.auth.update_user.assert_called_once_with(
            {"password": "password-123"}
        )

    def test_update_password_provider_failure_is_generic_and_chained(self):
        provider_error = RuntimeError("provider update detail")
        self.client.auth.update_user.side_effect = provider_error

        with self.assertRaises(AuthError) as raised:
            self.service.update_password("password-123", "password-123")

        self.assertEqual(
            str(raised.exception), "Password could not be updated. Try again."
        )
        self.assertNotIn("provider update detail", str(raised.exception))
        self.assertIs(raised.exception.__cause__, provider_error)

    def test_sign_out_calls_provider(self):
        self.service.sign_out()

        self.client.auth.sign_out.assert_called_once_with()

    def test_sign_out_provider_failure_is_generic_and_chained(self):
        provider_error = RuntimeError("provider logout detail")
        self.client.auth.sign_out.side_effect = provider_error

        with self.assertRaises(AuthError) as raised:
            self.service.sign_out()

        self.assertEqual(str(raised.exception), "Sign out could not be completed.")
        self.assertNotIn("provider logout detail", str(raised.exception))
        self.assertIs(raised.exception.__cause__, provider_error)

    def test_login_exception_becomes_generic_error(self):
        self.client.auth.sign_in_with_password.side_effect = RuntimeError(
            "user does not exist"
        )

        with self.assertRaisesRegex(AuthError, "Email or password is invalid"):
            self.service.sign_in("person@example.com", "password-123")


if __name__ == "__main__":
    unittest.main()
