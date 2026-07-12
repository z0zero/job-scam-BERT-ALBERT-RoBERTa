from dataclasses import dataclass
from typing import Any, Mapping

from supabase import Client, create_client


class SupabaseConfigError(RuntimeError):
    """Raised when required non-secret configuration is unavailable."""


@dataclass(frozen=True)
class SupabaseSettings:
    url: str
    publishable_key: str
    app_url: str


_REQUIRED_KEYS = (
    "SUPABASE_URL",
    "SUPABASE_PUBLISHABLE_KEY",
    "APP_URL",
)


def load_supabase_settings(secrets: Mapping[str, Any]) -> SupabaseSettings:
    values: dict[str, str] = {}
    missing: list[str] = []
    for key in _REQUIRED_KEYS:
        value = str(secrets.get(key, "")).strip()
        if not value:
            missing.append(key)
        values[key] = value

    if missing:
        raise SupabaseConfigError(
            "Missing Streamlit secret(s): " + ", ".join(missing)
        )

    return SupabaseSettings(
        url=values["SUPABASE_URL"],
        publishable_key=values["SUPABASE_PUBLISHABLE_KEY"],
        app_url=values["APP_URL"].rstrip("/"),
    )


def create_session_client(settings: SupabaseSettings) -> Client:
    """Create one mutable auth client for one Streamlit script run/session."""
    return create_client(settings.url, settings.publishable_key)
