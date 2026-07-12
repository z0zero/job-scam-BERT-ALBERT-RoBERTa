import unittest
from unittest.mock import patch


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class AuthViewTests(unittest.TestCase):
    def _render_auth_page(self, submitted_label, inputs):
        from src.views.auth_view import AuthView

        with (
            patch("src.views.auth_view.st.subheader"),
            patch(
                "src.views.auth_view.st.tabs",
                return_value=(_Context(), _Context(), _Context()),
            ),
            patch("src.views.auth_view.st.form", return_value=_Context()),
            patch(
                "src.views.auth_view.st.text_input",
                side_effect=lambda _label, **kwargs: inputs[kwargs["key"]],
            ),
            patch(
                "src.views.auth_view.st.form_submit_button",
                side_effect=lambda label, **_kwargs: label == submitted_label,
            ),
        ):
            return AuthView.render_auth_page()

    def test_auth_page_returns_login_action(self):
        action = self._render_auth_page(
            "Login",
            {
                "login_email": "user@example.com",
                "login_password": "secret",
            },
        )

        self.assertEqual(action.kind, "login")
        self.assertEqual(
            action.payload,
            {"email": "user@example.com", "password": "secret"},
        )

    def test_auth_page_returns_signup_action(self):
        action = self._render_auth_page(
            "Create account",
            {
                "login_email": "",
                "login_password": "",
                "signup_name": "Test User",
                "signup_email": "user@example.com",
                "signup_password": "password",
                "signup_confirmation": "password",
            },
        )

        self.assertEqual(action.kind, "signup")
        self.assertEqual(action.payload["full_name"], "Test User")
        self.assertEqual(action.payload["confirmation"], "password")

    def test_auth_page_returns_forgot_password_action(self):
        action = self._render_auth_page(
            "Send reset email",
            {
                "login_email": "",
                "login_password": "",
                "signup_name": "",
                "signup_email": "",
                "signup_password": "",
                "signup_confirmation": "",
                "forgot_email": "user@example.com",
            },
        )

        self.assertEqual(action.kind, "forgot_password")
        self.assertEqual(action.payload, {"email": "user@example.com"})

    def test_recovery_form_returns_update_password_action(self):
        from src.views.auth_view import AuthView

        values = {
            "new_password": "new-password",
            "new_password_confirmation": "new-password",
        }
        with (
            patch("src.views.auth_view.st.subheader"),
            patch("src.views.auth_view.st.form", return_value=_Context()),
            patch(
                "src.views.auth_view.st.text_input",
                side_effect=lambda _label, **kwargs: values[kwargs["key"]],
            ),
            patch(
                "src.views.auth_view.st.form_submit_button", return_value=True
            ),
        ):
            action = AuthView.render_recovery_form()

        self.assertEqual(action.kind, "update_password")
        self.assertEqual(
            action.payload,
            {
                "password": "new-password",
                "confirmation": "new-password",
            },
        )

    def test_feedback_helpers_delegate_to_streamlit(self):
        from src.views.auth_view import AuthView

        with (
            patch("src.views.auth_view.st.success") as success,
            patch("src.views.auth_view.st.error") as error,
        ):
            AuthView.render_success("Done")
            AuthView.render_error("Failed")

        success.assert_called_once_with("Done")
        error.assert_called_once_with("Failed")


if __name__ == "__main__":
    unittest.main()
