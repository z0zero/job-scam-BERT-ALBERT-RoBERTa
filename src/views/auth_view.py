from dataclasses import dataclass
from typing import Literal

import streamlit as st


AuthActionKind = Literal[
    "login", "signup", "forgot_password", "update_password"
]


@dataclass(frozen=True)
class AuthAction:
    kind: AuthActionKind
    payload: dict[str, str]


class AuthView:
    @staticmethod
    def render_auth_page() -> AuthAction | None:
        st.subheader("Account access")
        login_tab, signup_tab, forgot_tab = st.tabs(
            ["Login", "Sign up", "Forgot password"]
        )

        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email", key="login_email")
                password = st.text_input(
                    "Password", type="password", key="login_password"
                )
                if st.form_submit_button(
                    "Login", type="primary", use_container_width=True
                ):
                    return AuthAction(
                        "login", {"email": email, "password": password}
                    )

        with signup_tab:
            with st.form("signup_form"):
                full_name = st.text_input("Full name", key="signup_name")
                email = st.text_input("Email", key="signup_email")
                password = st.text_input(
                    "Password", type="password", key="signup_password"
                )
                confirmation = st.text_input(
                    "Confirm password",
                    type="password",
                    key="signup_confirmation",
                )
                if st.form_submit_button(
                    "Create account", type="primary", use_container_width=True
                ):
                    return AuthAction(
                        "signup",
                        {
                            "full_name": full_name,
                            "email": email,
                            "password": password,
                            "confirmation": confirmation,
                        },
                    )

        with forgot_tab:
            with st.form("forgot_password_form"):
                email = st.text_input("Email", key="forgot_email")
                if st.form_submit_button(
                    "Send reset email", use_container_width=True
                ):
                    return AuthAction("forgot_password", {"email": email})
        return None

    @staticmethod
    def render_recovery_form() -> AuthAction | None:
        st.subheader("Choose a new password")
        with st.form("update_password_form"):
            password = st.text_input(
                "New password", type="password", key="new_password"
            )
            confirmation = st.text_input(
                "Confirm new password",
                type="password",
                key="new_password_confirmation",
            )
            if st.form_submit_button(
                "Update password", type="primary", use_container_width=True
            ):
                return AuthAction(
                    "update_password",
                    {"password": password, "confirmation": confirmation},
                )
        return None

    @staticmethod
    def render_success(message: str) -> None:
        st.success(message)

    @staticmethod
    def render_error(message: str) -> None:
        st.error(message)
