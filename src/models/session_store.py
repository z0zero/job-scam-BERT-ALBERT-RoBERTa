from collections.abc import MutableMapping
from typing import Any

from src.models.auth_service import AuthSession, AuthenticatedUser


AUTH_STATE_KEY = "supabase_auth"
RECOVERY_MODE_KEY = "supabase_recovery_mode"
MODEL_LOADING_PENDING_KEY = "model_loading_pending"
AUTH_NOTICE_KEY = "auth_notice"

_AUTH_FIELDS = (
    "user_id",
    "email",
    "full_name",
    "access_token",
    "refresh_token",
)


def save_auth_session(
    state: MutableMapping[str, Any], session: AuthSession
) -> None:
    state[AUTH_STATE_KEY] = {
        "user_id": session.user.id,
        "email": session.user.email,
        "full_name": session.user.full_name,
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
    }


def load_auth_session(
    state: MutableMapping[str, Any],
) -> AuthSession | None:
    value = state.get(AUTH_STATE_KEY)
    if not isinstance(value, dict):
        return None
    if not all(isinstance(value.get(field), str) for field in _AUTH_FIELDS):
        return None
    return AuthSession(
        user=AuthenticatedUser(
            id=value["user_id"],
            email=value["email"],
            full_name=value["full_name"],
        ),
        access_token=value["access_token"],
        refresh_token=value["refresh_token"],
    )


def load_auth_tokens(
    state: MutableMapping[str, Any],
) -> tuple[str, str] | None:
    session = load_auth_session(state)
    if session is None:
        return None
    return session.access_token, session.refresh_token


def mark_recovery_mode(
    state: MutableMapping[str, Any], enabled: bool
) -> None:
    state[RECOVERY_MODE_KEY] = enabled


def is_recovery_mode(state: MutableMapping[str, Any]) -> bool:
    return state.get(RECOVERY_MODE_KEY) is True


def mark_model_loading_pending(
    state: MutableMapping[str, Any], enabled: bool
) -> None:
    if enabled is True:
        state[MODEL_LOADING_PENDING_KEY] = True
    else:
        state.pop(MODEL_LOADING_PENDING_KEY, None)


def is_model_loading_pending(state: MutableMapping[str, Any]) -> bool:
    return state.get(MODEL_LOADING_PENDING_KEY) is True


def set_auth_notice(
    state: MutableMapping[str, Any], message: str
) -> None:
    if isinstance(message, str) and message:
        state[AUTH_NOTICE_KEY] = message
    else:
        state.pop(AUTH_NOTICE_KEY, None)


def pop_auth_notice(
    state: MutableMapping[str, Any],
) -> str | None:
    value = state.pop(AUTH_NOTICE_KEY, None)
    if isinstance(value, str) and value:
        return value
    return None


def clear_auth_state(state: MutableMapping[str, Any]) -> None:
    state.pop(AUTH_STATE_KEY, None)
    state.pop(RECOVERY_MODE_KEY, None)
    state.pop(MODEL_LOADING_PENDING_KEY, None)
