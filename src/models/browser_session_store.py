from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import streamlit as st


_STORAGE_KEY = "job_scam_supabase_session"
_COMPONENT_KEY = "browser_auth_session"

_BROWSER_SESSION_COMPONENT = st.components.v2.component(
    "browser_session_store",
    js=f"""
    export default function({{ data, setStateValue }}) {{
        const storageKey = { _STORAGE_KEY!r };
        let tokens = null;

        try {{
            if (data.clear) {{
                window.sessionStorage.removeItem(storageKey);
            }} else if (data.tokens) {{
                tokens = data.tokens;
                window.sessionStorage.setItem(storageKey, JSON.stringify(tokens));
            }} else {{
                const stored = window.sessionStorage.getItem(storageKey);
                if (stored) {{
                    const parsed = JSON.parse(stored);
                    if (
                        typeof parsed?.access_token === "string" &&
                        typeof parsed?.refresh_token === "string"
                    ) {{
                        tokens = parsed;
                    }} else {{
                        window.sessionStorage.removeItem(storageKey);
                    }}
                }}
            }}
        }} catch (_) {{
            tokens = null;
        }}

        setStateValue("snapshot", {{
            fingerprint: data.fingerprint,
            tokens: data.clear ? null : tokens,
        }});
    }}
    """,
)


@dataclass(frozen=True)
class BrowserSessionSnapshot:
    ready: bool
    tokens: tuple[str, str] | None


class BrowserSessionStore:
    def __init__(self, renderer: Callable[..., Any] | None = None):
        self.renderer = renderer or _BROWSER_SESSION_COMPONENT

    def sync(
        self,
        tokens: tuple[str, str] | None,
        *,
        clear: bool = False,
    ) -> BrowserSessionSnapshot:
        valid_tokens = _valid_tokens(tokens)
        fingerprint = _fingerprint(valid_tokens, clear=clear)
        payload = None
        if valid_tokens is not None and not clear:
            payload = {
                "access_token": valid_tokens[0],
                "refresh_token": valid_tokens[1],
            }

        result = self.renderer(
            data={
                "clear": clear,
                "fingerprint": fingerprint,
                "tokens": payload,
            },
            default={"snapshot": None},
            key=_COMPONENT_KEY,
            on_snapshot_change=lambda: None,
        )
        snapshot = getattr(result, "snapshot", None)
        if not isinstance(snapshot, Mapping):
            return BrowserSessionSnapshot(False, None)
        if snapshot.get("fingerprint") != fingerprint:
            return BrowserSessionSnapshot(False, None)
        if clear:
            return BrowserSessionSnapshot(True, None)

        browser_tokens = snapshot.get("tokens")
        if not isinstance(browser_tokens, Mapping):
            return BrowserSessionSnapshot(True, None)
        parsed = _valid_tokens(
            (
                browser_tokens.get("access_token"),
                browser_tokens.get("refresh_token"),
            )
        )
        return BrowserSessionSnapshot(True, parsed)


def _valid_tokens(tokens: Any) -> tuple[str, str] | None:
    if not isinstance(tokens, tuple) or len(tokens) != 2:
        return None
    access_token, refresh_token = tokens
    if not isinstance(access_token, str) or not access_token.strip():
        return None
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        return None
    return access_token, refresh_token


def _fingerprint(tokens: tuple[str, str] | None, *, clear: bool) -> str:
    if clear:
        return "clear"
    if tokens is None:
        return "read"
    digest = sha256(f"{tokens[0]}\0{tokens[1]}".encode()).hexdigest()
    return f"write:{digest}"
