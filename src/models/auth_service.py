import re
from dataclasses import dataclass
from typing import Any

from supabase import Client


class ValidationError(ValueError):
    """Safe validation message that can be shown to the user."""


class AuthError(RuntimeError):
    """Safe authentication message that does not expose provider internals."""


@dataclass(frozen=True)
class AuthenticatedUser:
    id: str
    email: str
    full_name: str


@dataclass(frozen=True)
class AuthSession:
    user: AuthenticatedUser
    access_token: str
    refresh_token: str


_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _validate_email(email: str) -> str:
    normalized = normalize_email(email)
    if not _EMAIL_PATTERN.fullmatch(normalized):
        raise ValidationError("Enter a valid email address.")
    return normalized


def validate_new_password(password: str, confirmation: str) -> None:
    if len(password) < 8:
        raise ValidationError("Password must contain at least 8 characters.")
    if password != confirmation:
        raise ValidationError("Password confirmation does not match.")


def validate_signup(
    full_name: str, email: str, password: str, confirmation: str
) -> tuple[str, str]:
    cleaned_name = full_name.strip()
    if not cleaned_name:
        raise ValidationError("Full name is required.")
    normalized_email = _validate_email(email)
    validate_new_password(password, confirmation)
    return cleaned_name, normalized_email


class AuthService:
    def __init__(self, client: Client, app_url: str):
        self.client = client
        self.app_url = app_url

    def sign_up(
        self,
        full_name: str,
        email: str,
        password: str,
        confirmation: str,
    ) -> None:
        cleaned_name, normalized_email = validate_signup(
            full_name, email, password, confirmation
        )
        try:
            self.client.auth.sign_up(
                {
                    "email": normalized_email,
                    "password": password,
                    "options": {
                        "data": {"full_name": cleaned_name},
                        "email_redirect_to": self.app_url,
                    },
                }
            )
        except Exception as exc:
            raise AuthError(
                "Sign up could not be completed. Check the form and try again."
            ) from exc

    def sign_in(self, email: str, password: str) -> AuthSession:
        normalized_email = _validate_email(email)
        if not password:
            raise ValidationError("Password is required.")
        try:
            response = self.client.auth.sign_in_with_password(
                {"email": normalized_email, "password": password}
            )
            return self._to_session(response)
        except ValidationError:
            raise
        except Exception as exc:
            raise AuthError("Email or password is invalid.") from exc

    def restore_session(
        self, access_token: str, refresh_token: str
    ) -> AuthSession:
        if not access_token.strip():
            raise ValidationError("Access token is missing.")
        if not refresh_token.strip():
            raise ValidationError("Refresh token is missing.")
        try:
            response = self.client.auth.set_session(access_token, refresh_token)
            return self._to_session(response)
        except Exception as exc:
            raise AuthError("Your session has expired. Please sign in again.") from exc

    def verify_token(self, token_hash: str, otp_type: str) -> AuthSession:
        if otp_type not in {"email", "recovery"}:
            raise ValidationError("Unsupported authentication callback.")
        if not token_hash.strip():
            raise ValidationError("Authentication token is missing.")
        try:
            response = self.client.auth.verify_otp(
                {"token_hash": token_hash, "type": otp_type}
            )
            return self._to_session(response)
        except Exception as exc:
            raise AuthError(
                "This authentication link is invalid or has expired."
            ) from exc

    def request_password_reset(self, email: str) -> None:
        normalized_email = _validate_email(email)
        try:
            self.client.auth.reset_password_for_email(
                normalized_email, {"redirect_to": self.app_url}
            )
        except Exception as exc:
            raise AuthError(
                "If an account exists for that email, a reset message will be sent."
            ) from exc

    def update_password(self, password: str, confirmation: str) -> None:
        validate_new_password(password, confirmation)
        try:
            self.client.auth.update_user({"password": password})
        except Exception as exc:
            raise AuthError("Password could not be updated. Try again.") from exc

    def sign_out(self) -> None:
        try:
            self.client.auth.sign_out()
        except Exception as exc:
            raise AuthError("Sign out could not be completed.") from exc

    @staticmethod
    def _to_session(response: Any) -> AuthSession:
        session = getattr(response, "session", None)
        user = getattr(session, "user", None) or getattr(response, "user", None)
        if session is None or user is None:
            raise AuthError("Supabase did not return a valid session.")
        metadata = getattr(user, "user_metadata", None) or {}
        email = getattr(user, "email", "") or ""
        return AuthSession(
            user=AuthenticatedUser(
                id=str(user.id),
                email=email,
                full_name=str(metadata.get("full_name") or email),
            ),
            access_token=session.access_token,
            refresh_token=session.refresh_token,
        )
