# Supabase Authentication and Analysis History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add mandatory Supabase email/password authentication, email verification and recovery, isolated Streamlit sessions, and per-user read-only analysis history to the Job Scam Detection app.

**Architecture:** Preserve the existing MVC split. A session-scoped Supabase client feeds an `AuthService` and `HistoryRepository`; `AppController` enforces the auth gate before classifier loading and coordinates dumb Streamlit views. Supabase Auth owns identities and password flows, while a single RLS-protected Postgres table owns analysis history.

**Tech Stack:** Python 3.12.1, Streamlit, `supabase==2.31.0`, Supabase Auth, Supabase Postgres, PostgREST, PostgreSQL RLS, `unittest`

**Spec:** `docs/superpowers/specs/2026-07-12-supabase-authentication-history-design.md`

## Global Constraints

- Use only `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`, and `APP_URL` from Streamlit Secrets; never use `service_role`, a secret key, or a database password in application code.
- Never decorate a Supabase client or auth service with `st.cache_resource`; auth state must remain isolated per Streamlit session.
- Store access and refresh tokens only in `st.session_state`, clear them locally even if remote sign-out fails, and do not implement remember-me persistence.
- Enforce full name, normalized email, password confirmation, and a minimum password length of 8 characters before calling Supabase.
- Keep the existing 1,500-word analysis limit and existing image size/type validation.
- Persist the complete analyzed text, `text`/`image` source, classifier label, confidence in `[0, 1]`, red flags, and database timestamp; never persist the uploaded image.
- Fetch at most 20 history rows per request and expose no application update/delete operation.
- Enable RLS and combine `TO authenticated` with `(select auth.uid()) = user_id`; `TO authenticated` alone is insufficient authorization.
- Grant only `SELECT` and `INSERT` on `public.analysis_history` to `authenticated`; grant nothing to `anon` and never add update/delete policies.
- Keep user metadata such as `full_name` out of authorization decisions.
- Use Supabase built-in SMTP only for thesis-demo team addresses; custom SMTP, CAPTCHA, trusted redirect allowlists, SPF, DKIM, and DMARC are production prerequisites, not part of this implementation.
- Keep existing UI copy in English to match the current application.
- Sign every implementation commit with the configured GPG key using `git commit -S`.

---

## File Map

| File | Responsibility |
| --- | --- |
| `requirements.txt` | Add the exact supported Supabase Python client version. |
| `.gitignore` | Exclude the real Streamlit secrets file. |
| `.streamlit/secrets.example.toml` | Safe copyable configuration shape with deliberately fake values. |
| `src/models/supabase_client.py` | Validate secrets and create a fresh session-scoped Supabase client. |
| `src/models/auth_service.py` | Validate input and wrap all Supabase Auth operations behind stable dataclasses/errors. |
| `src/models/session_store.py` | Serialize, restore, mark recovery, and clear auth state in `st.session_state`. |
| `supabase/migrations/20260712150000_create_analysis_history.sql` | Create the constrained history table, index, grants, and RLS policies. |
| `supabase/tests/analysis_history_rls.sql` | Transactional verification of two-user isolation and denied operations. |
| `src/models/history_repository.py` | Validate history payloads and perform paginated insert/select calls. |
| `src/views/auth_view.py` | Render login, sign-up, forgot-password, and new-password forms as actions. |
| `src/views/history_view.py` | Render 20-row history pages, navigation, empty/error states, and details. |
| `src/views/main_view.py` | Return the input source and render authenticated sidebar navigation. |
| `src/controllers/app_controller.py` | Enforce auth/callback gates, dispatch auth actions, run analysis, persist history, and handle navigation/logout. |
| `tests/models/test_supabase_client.py` | Configuration and client-factory unit tests. |
| `tests/models/test_auth_service.py` | Auth validation, payload, session conversion, refresh, recovery, and safe-error tests. |
| `tests/models/test_session_store.py` | Session save/load/recovery/clear unit tests. |
| `tests/models/test_history_repository.py` | Payload validation, insert, pagination, and query-scoping unit tests. |
| `tests/views/test_history_view.py` | Pure history formatting helper tests. |
| `tests/controllers/test_app_controller.py` | Auth gate, classifier loading, persistence failure, and logout tests. |
| `docs/supabase-setup.md` | Dashboard, migration, email-template, demo SMTP, and deployment instructions. |
| `README.md` / `README.id.md` | User-facing setup and feature summaries. |

---

### Task 1: Supabase Configuration and Session-Scoped Client

**Files:**
- Modify: `requirements.txt:11`
- Modify: `.gitignore:14`
- Create: `.streamlit/secrets.example.toml`
- Create: `src/models/supabase_client.py`
- Create: `tests/__init__.py`
- Create: `tests/models/__init__.py`
- Create: `tests/models/test_supabase_client.py`

**Interfaces:**
- Consumes: a mapping compatible with `st.secrets`.
- Produces: `SupabaseSettings`, `SupabaseConfigError`, `load_supabase_settings(secrets)`, and `create_session_client(settings)`.

- [ ] **Step 1: Add the approved dependency and safe secrets template**

Append this exact dependency to `requirements.txt`:

```txt
supabase==2.31.0
```

Append this exact ignore rule to `.gitignore`:

```gitignore
.streamlit/secrets.toml
```

Create `.streamlit/secrets.example.toml`:

```toml
SUPABASE_URL = "https://example.supabase.co"
SUPABASE_PUBLISHABLE_KEY = "sb_publishable_example"
APP_URL = "http://localhost:8501"
```

- [ ] **Step 2: Install the pinned dependency**

Run:

```powershell
.\venv\Scripts\python.exe -m pip install "supabase==2.31.0"
```

Expected: pip installs `supabase 2.31.0`; do not install the `3.0.0a1` prerelease.

- [ ] **Step 3: Write failing configuration tests**

Create `tests/__init__.py` and `tests/models/__init__.py` as empty files. Create `tests/models/test_supabase_client.py`:

```python
import unittest
from unittest.mock import patch

from src.models.supabase_client import (
    SupabaseConfigError,
    SupabaseSettings,
    create_session_client,
    load_supabase_settings,
)


class SupabaseClientTests(unittest.TestCase):
    def test_load_settings_strips_values(self):
        settings = load_supabase_settings(
            {
                "SUPABASE_URL": " https://project.supabase.co ",
                "SUPABASE_PUBLISHABLE_KEY": " sb_publishable_test ",
                "APP_URL": " http://localhost:8501 ",
            }
        )

        self.assertEqual(
            settings,
            SupabaseSettings(
                url="https://project.supabase.co",
                publishable_key="sb_publishable_test",
                app_url="http://localhost:8501",
            ),
        )

    def test_load_settings_names_missing_secret_without_printing_values(self):
        with self.assertRaisesRegex(
            SupabaseConfigError, "SUPABASE_PUBLISHABLE_KEY"
        ):
            load_supabase_settings(
                {
                    "SUPABASE_URL": "https://project.supabase.co",
                    "APP_URL": "http://localhost:8501",
                }
            )

    @patch("src.models.supabase_client.create_client")
    def test_create_session_client_uses_publishable_key(self, create_client):
        settings = SupabaseSettings(
            url="https://project.supabase.co",
            publishable_key="sb_publishable_test",
            app_url="http://localhost:8501",
        )

        result = create_session_client(settings)

        self.assertIs(result, create_client.return_value)
        create_client.assert_called_once_with(
            "https://project.supabase.co", "sb_publishable_test"
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run the test to verify it fails**

Run:

```powershell
.\venv\Scripts\python.exe -m unittest tests.models.test_supabase_client -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.supabase_client'`.

- [ ] **Step 5: Implement the configuration boundary**

Create `src/models/supabase_client.py`:

```python
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
```

- [ ] **Step 6: Run the focused tests**

Run:

```powershell
.\venv\Scripts\python.exe -m unittest tests.models.test_supabase_client -v
```

Expected: 3 tests PASS.

- [ ] **Step 7: Commit the configuration boundary**

```powershell
git add requirements.txt .gitignore .streamlit/secrets.example.toml src/models/supabase_client.py tests/__init__.py tests/models/__init__.py tests/models/test_supabase_client.py
git commit -S -m "feat: add Supabase client configuration"
```

---

### Task 2: Authentication Service and Safe Validation

**Files:**
- Create: `src/models/auth_service.py`
- Create: `tests/models/test_auth_service.py`

**Interfaces:**
- Consumes: a session-scoped Supabase `Client` and exact `APP_URL`.
- Produces: `AuthenticatedUser`, `AuthSession`, `AuthError`, `ValidationError`, `normalize_email`, `validate_signup`, `validate_new_password`, and `AuthService` methods `sign_up`, `sign_in`, `restore_session`, `verify_token`, `request_password_reset`, `update_password`, and `sign_out`.

- [ ] **Step 1: Write failing auth tests**

Create `tests/models/test_auth_service.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
.\venv\Scripts\python.exe -m unittest tests.models.test_auth_service -v
```

Expected: FAIL because `src.models.auth_service` does not exist.

- [ ] **Step 3: Implement the auth service**

Create `src/models/auth_service.py`:

```python
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
        try:
            response = self.client.auth.set_session(access_token, refresh_token)
            return self._to_session(response)
        except Exception as exc:
            raise AuthError("Your session has expired. Please sign in again.") from exc

    def verify_token(self, token_hash: str, otp_type: str) -> AuthSession:
        if otp_type not in {"email", "recovery"}:
            raise ValidationError("Unsupported authentication callback.")
        if not token_hash:
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
        self.client.auth.sign_out()

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
```

- [ ] **Step 4: Run the focused auth tests**

Run:

```powershell
.\venv\Scripts\python.exe -m unittest tests.models.test_auth_service -v
```

Expected: 8 tests PASS.

- [ ] **Step 5: Commit the auth boundary**

```powershell
git add src/models/auth_service.py tests/models/test_auth_service.py
git commit -S -m "feat: add Supabase authentication service"
```

---

### Task 3: Streamlit Session State Store

**Files:**
- Create: `src/models/session_store.py`
- Create: `tests/models/test_session_store.py`

**Interfaces:**
- Consumes: `MutableMapping[str, Any]` and `AuthSession`.
- Produces: `save_auth_session`, `load_auth_session`, `load_auth_tokens`, `mark_recovery_mode`, `is_recovery_mode`, and `clear_auth_state`.

- [ ] **Step 1: Write failing session-store tests**

Create `tests/models/test_session_store.py`:

```python
import unittest

from src.models.auth_service import AuthSession, AuthenticatedUser
from src.models.session_store import (
    clear_auth_state,
    is_recovery_mode,
    load_auth_session,
    load_auth_tokens,
    mark_recovery_mode,
    save_auth_session,
)


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.state = {}
        self.session = AuthSession(
            user=AuthenticatedUser(
                id="user-a", email="person@example.com", full_name="Person A"
            ),
            access_token="access-a",
            refresh_token="refresh-a",
        )

    def test_save_and_load_round_trip(self):
        save_auth_session(self.state, self.session)
        self.assertEqual(load_auth_session(self.state), self.session)
        self.assertEqual(load_auth_tokens(self.state), ("access-a", "refresh-a"))

    def test_recovery_marker_is_explicit(self):
        self.assertFalse(is_recovery_mode(self.state))
        mark_recovery_mode(self.state, True)
        self.assertTrue(is_recovery_mode(self.state))

    def test_clear_removes_tokens_identity_and_recovery(self):
        save_auth_session(self.state, self.session)
        mark_recovery_mode(self.state, True)
        clear_auth_state(self.state)
        self.assertIsNone(load_auth_session(self.state))
        self.assertIsNone(load_auth_tokens(self.state))
        self.assertFalse(is_recovery_mode(self.state))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
.\venv\Scripts\python.exe -m unittest tests.models.test_session_store -v
```

Expected: FAIL because `src.models.session_store` does not exist.

- [ ] **Step 3: Implement session serialization**

Create `src/models/session_store.py`:

```python
from collections.abc import MutableMapping
from typing import Any

from src.models.auth_service import AuthSession, AuthenticatedUser


AUTH_STATE_KEY = "supabase_auth"
RECOVERY_MODE_KEY = "supabase_recovery_mode"


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
    required = {
        "user_id",
        "email",
        "full_name",
        "access_token",
        "refresh_token",
    }
    if not required.issubset(value):
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


def clear_auth_state(state: MutableMapping[str, Any]) -> None:
    state.pop(AUTH_STATE_KEY, None)
    state.pop(RECOVERY_MODE_KEY, None)
```

- [ ] **Step 4: Run focused tests**

```powershell
.\venv\Scripts\python.exe -m unittest tests.models.test_session_store -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit the session store**

```powershell
git add src/models/session_store.py tests/models/test_session_store.py
git commit -S -m "feat: isolate Supabase session state"
```

---

### Task 4: History Schema, RLS Verification, and Repository

**Files:**
- Create: `supabase/migrations/20260712150000_create_analysis_history.sql`
- Create: `supabase/tests/analysis_history_rls.sql`
- Create: `src/models/history_repository.py`
- Create: `tests/models/test_history_repository.py`

**Interfaces:**
- Consumes: authenticated Supabase client and `AnalysisHistoryCreate`.
- Produces: `HistoryError`, `AnalysisHistoryCreate`, `HistoryPage`, `HistoryRepository.create`, and `HistoryRepository.list_page`.

- [ ] **Step 1: Create the migration through the Supabase CLI migration workflow**

The Supabase CLI is not currently installed. At execution time, request approval to install/use it, then run `supabase migration new --help` to verify current syntax followed by:

```powershell
supabase migration new create_analysis_history
```

Rename the newly generated empty migration to the repository-stable exact path `supabase/migrations/20260712150000_create_analysis_history.sql` before adding SQL. Do not apply DDL to a production project while iterating.

Use this PowerShell command immediately after `supabase migration new`:

```powershell
$generated = Get-ChildItem -LiteralPath 'supabase\migrations' -Filter '*_create_analysis_history.sql' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Move-Item -LiteralPath $generated.FullName -Destination 'supabase\migrations\20260712150000_create_analysis_history.sql'
```

- [ ] **Step 2: Write the constrained schema and RLS policies**

Put this exact SQL in `supabase/migrations/20260712150000_create_analysis_history.sql`:

```sql
create table public.analysis_history (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    input_text text not null check (length(btrim(input_text)) > 0),
    input_source text not null check (input_source in ('text', 'image')),
    prediction_label text not null check (
        prediction_label in ('Legitimate Job', 'Potential Scam')
    ),
    confidence double precision not null check (
        confidence >= 0.0 and confidence <= 1.0
    ),
    red_flags jsonb not null default '[]'::jsonb check (
        jsonb_typeof(red_flags) = 'array'
    ),
    created_at timestamptz not null default now()
);

create index analysis_history_user_created_idx
    on public.analysis_history (user_id, created_at desc);

alter table public.analysis_history enable row level security;

revoke all on table public.analysis_history from anon;
revoke all on table public.analysis_history from authenticated;
grant select, insert on table public.analysis_history to authenticated;

create policy "Users can read their own analysis history"
on public.analysis_history
for select
to authenticated
using ((select auth.uid()) = user_id);

create policy "Users can insert their own analysis history"
on public.analysis_history
for insert
to authenticated
with check ((select auth.uid()) = user_id);
```

- [ ] **Step 3: Add a transactional RLS verification script**

Create `supabase/tests/analysis_history_rls.sql`. It requires two verified users in the development project and rolls back all test rows:

```sql
begin;

do $$
declare
    user_a uuid;
    user_b uuid;
begin
    select id into user_a from auth.users order by created_at limit 1;
    select id into user_b from auth.users order by created_at offset 1 limit 1;
    if user_a is null or user_b is null then
        raise exception 'RLS test requires two verified auth users';
    end if;
    perform set_config('test.user_a', user_a::text, true);
    perform set_config('test.user_b', user_b::text, true);
    insert into public.analysis_history (
        user_id, input_text, input_source, prediction_label, confidence, red_flags
    ) values
        (user_a, 'owned by A', 'text', 'Legitimate Job', 0.91, '[]'),
        (user_b, 'owned by B', 'text', 'Potential Scam', 0.88, '["flag"]');
end $$;

set local role authenticated;
select set_config(
    'request.jwt.claims',
    json_build_object(
        'sub', current_setting('test.user_a'),
        'role', 'authenticated'
    )::text,
    true
);

do $$
declare
    visible_count integer;
    rejected boolean := false;
begin
    select count(*) into visible_count from public.analysis_history;
    if visible_count <> 1 then
        raise exception 'User A should see exactly one owned row, saw %', visible_count;
    end if;

    begin
        insert into public.analysis_history (
            user_id, input_text, input_source,
            prediction_label, confidence, red_flags
        ) values (
            current_setting('test.user_b')::uuid,
            'forged owner', 'text', 'Potential Scam', 0.5, '[]'
        );
    exception when insufficient_privilege then
        rejected := true;
    end;
    if not rejected then
        raise exception 'Cross-user insert was not rejected';
    end if;

    rejected := false;
    begin
        update public.analysis_history set confidence = 0.1;
    exception when insufficient_privilege then
        rejected := true;
    end;
    if not rejected then
        raise exception 'UPDATE was not rejected';
    end if;

    rejected := false;
    begin
        delete from public.analysis_history;
    exception when insufficient_privilege then
        rejected := true;
    end;
    if not rejected then
        raise exception 'DELETE was not rejected';
    end if;
end $$;

reset role;
set local role anon;
select set_config(
    'request.jwt.claims',
    json_build_object('role', 'anon')::text,
    true
);

do $$
declare
    rejected boolean := false;
begin
    begin
        perform count(*) from public.analysis_history;
    exception when insufficient_privilege then
        rejected := true;
    end;
    if not rejected then
        raise exception 'Anonymous SELECT was not rejected';
    end if;
end $$;

reset role;
rollback;
```

- [ ] **Step 4: Write failing repository tests**

Create `tests/models/test_history_repository.py`:

```python
import unittest
from types import SimpleNamespace

from src.models.history_repository import (
    AnalysisHistoryCreate,
    HistoryError,
    HistoryRepository,
)


class FakeQuery:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def insert(self, payload):
        self.calls.append(("insert", payload))
        return self

    def select(self, columns):
        self.calls.append(("select", columns))
        return self

    def eq(self, column, value):
        self.calls.append(("eq", column, value))
        return self

    def order(self, column, desc=False):
        self.calls.append(("order", column, desc))
        return self

    def range(self, start, end):
        self.calls.append(("range", start, end))
        return self

    def execute(self):
        return SimpleNamespace(data=self.data)


class FakeClient:
    def __init__(self, data):
        self.query = FakeQuery(data)

    def table(self, name):
        self.query.calls.append(("table", name))
        return self.query


class HistoryRepositoryTests(unittest.TestCase):
    def test_create_scopes_payload_to_authenticated_user(self):
        client = FakeClient([{"id": "history-1"}])
        repository = HistoryRepository(client)
        record = AnalysisHistoryCreate(
            user_id="user-a",
            input_text="Job description",
            input_source="text",
            prediction_label="Legitimate Job",
            confidence=0.92,
            red_flags=[],
        )

        result = repository.create(record)

        self.assertEqual(result["id"], "history-1")
        insert_call = next(call for call in client.query.calls if call[0] == "insert")
        self.assertEqual(insert_call[1]["user_id"], "user-a")

    def test_create_rejects_out_of_range_confidence(self):
        repository = HistoryRepository(FakeClient([]))
        with self.assertRaises(HistoryError):
            repository.create(
                AnalysisHistoryCreate(
                    user_id="user-a",
                    input_text="Job",
                    input_source="text",
                    prediction_label="Legitimate Job",
                    confidence=1.1,
                    red_flags=[],
                )
            )

    def test_list_page_fetches_one_extra_row_to_detect_next_page(self):
        rows = [{"id": str(index)} for index in range(21)]
        client = FakeClient(rows)
        page = HistoryRepository(client).list_page("user-a", offset=20)

        self.assertEqual(len(page.items), 20)
        self.assertTrue(page.has_more)
        self.assertIn(("eq", "user_id", "user-a"), client.query.calls)
        self.assertIn(("range", 20, 40), client.query.calls)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 5: Run repository tests to verify they fail**

```powershell
.\venv\Scripts\python.exe -m unittest tests.models.test_history_repository -v
```

Expected: FAIL because `src.models.history_repository` does not exist.

- [ ] **Step 6: Implement the repository**

Create `src/models/history_repository.py`:

```python
from dataclasses import dataclass
from typing import Any

from supabase import Client


class HistoryError(RuntimeError):
    """Safe persistence error for analysis history."""


@dataclass(frozen=True)
class AnalysisHistoryCreate:
    user_id: str
    input_text: str
    input_source: str
    prediction_label: str
    confidence: float
    red_flags: list[str]


@dataclass(frozen=True)
class HistoryPage:
    items: list[dict[str, Any]]
    has_more: bool


class HistoryRepository:
    PAGE_SIZE = 20

    def __init__(self, client: Client):
        self.client = client

    def create(self, record: AnalysisHistoryCreate) -> dict[str, Any]:
        self._validate(record)
        payload = {
            "user_id": record.user_id,
            "input_text": record.input_text,
            "input_source": record.input_source,
            "prediction_label": record.prediction_label,
            "confidence": record.confidence,
            "red_flags": record.red_flags,
        }
        try:
            response = (
                self.client.table("analysis_history")
                .insert(payload)
                .execute()
            )
        except Exception as exc:
            raise HistoryError("Analysis history could not be saved.") from exc
        if not response.data:
            raise HistoryError("Analysis history could not be saved.")
        return response.data[0]

    def list_page(self, user_id: str, offset: int = 0) -> HistoryPage:
        if not user_id:
            raise HistoryError("Authenticated user is required.")
        if offset < 0 or offset % self.PAGE_SIZE != 0:
            raise HistoryError("Invalid history offset.")
        try:
            response = (
                self.client.table("analysis_history")
                .select(
                    "id,input_text,input_source,prediction_label,"
                    "confidence,red_flags,created_at"
                )
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .range(offset, offset + self.PAGE_SIZE)
                .execute()
            )
        except Exception as exc:
            raise HistoryError("Analysis history could not be loaded.") from exc
        rows = list(response.data or [])
        return HistoryPage(
            items=rows[: self.PAGE_SIZE],
            has_more=len(rows) > self.PAGE_SIZE,
        )

    @staticmethod
    def _validate(record: AnalysisHistoryCreate) -> None:
        if not record.user_id:
            raise HistoryError("Authenticated user is required.")
        if not record.input_text.strip():
            raise HistoryError("Analysis text is required.")
        if record.input_source not in {"text", "image"}:
            raise HistoryError("Invalid analysis input source.")
        if record.prediction_label not in {"Legitimate Job", "Potential Scam"}:
            raise HistoryError("Invalid prediction label.")
        if not 0.0 <= record.confidence <= 1.0:
            raise HistoryError("Confidence must be between 0 and 1.")
        if not all(isinstance(flag, str) for flag in record.red_flags):
            raise HistoryError("Red flags must be strings.")
```

- [ ] **Step 7: Run repository tests**

```powershell
.\venv\Scripts\python.exe -m unittest tests.models.test_history_repository -v
```

Expected: 3 tests PASS.

- [ ] **Step 8: Validate the migration without touching production**

Run the current CLI help first, then start local Supabase and reset the disposable local database:

```powershell
supabase db reset --help
supabase start
supabase db reset
```

Expected: migration applies successfully. Create two local verified users, execute `supabase/tests/analysis_history_rls.sql` in local Studio, and expect the transaction to finish with no raised exception and `ROLLBACK`.

Use local Studio at `http://localhost:54323`: open **Authentication → Users → Add user**, create and auto-confirm `rls-a@example.test` and `rls-b@example.test` with distinct passwords of at least eight characters, then open **SQL Editor**, paste the complete committed `supabase/tests/analysis_history_rls.sql`, and run it. Expected final command status: `ROLLBACK`, with no exception message.

- [ ] **Step 9: Commit persistence and RLS**

```powershell
git add supabase/migrations/20260712150000_create_analysis_history.sql supabase/tests/analysis_history_rls.sql src/models/history_repository.py tests/models/test_history_repository.py
git commit -S -m "feat: add RLS-protected analysis history"
```

---

### Task 5: Authentication View Actions

**Files:**
- Create: `src/views/auth_view.py`

**Interfaces:**
- Consumes: no service; renders from Streamlit only.
- Produces: `AuthAction(kind, payload)`, `AuthView.render_auth_page`, `AuthView.render_recovery_form`, `AuthView.render_success`, and `AuthView.render_error`.

- [ ] **Step 1: Implement declarative auth forms**

Create `src/views/auth_view.py`:

```python
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
```

- [ ] **Step 2: Compile the view**

```powershell
.\venv\Scripts\python.exe -m compileall -q src/views/auth_view.py
```

Expected: exit code 0.

- [ ] **Step 3: Commit the auth view**

```powershell
git add src/views/auth_view.py
git commit -S -m "feat: add authentication forms"
```

---

### Task 6: History View and Main View Contract

**Files:**
- Create: `src/views/history_view.py`
- Modify: `src/views/main_view.py:43-102`
- Create: `tests/views/__init__.py`
- Create: `tests/views/test_history_view.py`

**Interfaces:**
- Consumes: `HistoryPage.items`, current offset, current `AuthenticatedUser`.
- Produces: `make_snippet`, `format_confidence`, `HistoryView.render(items, offset, has_more) -> int | None`, `MainView.render_sidebar(user) -> tuple[str, bool]`, and `MainView.render_input_section(...) -> tuple[str, str, bool]`.

- [ ] **Step 1: Write failing pure formatting tests**

Create empty `tests/views/__init__.py` and create `tests/views/test_history_view.py`:

```python
import unittest

from src.views.history_view import format_confidence, make_snippet


class HistoryFormattingTests(unittest.TestCase):
    def test_make_snippet_collapses_whitespace_and_truncates(self):
        text = "  first\nsecond  " + ("x" * 250)
        result = make_snippet(text, limit=20)
        self.assertEqual(result, "first second xxxxxxx…")

    def test_format_confidence(self):
        self.assertEqual(format_confidence(0.9234), "92.3%")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
.\venv\Scripts\python.exe -m unittest tests.views.test_history_view -v
```

Expected: FAIL because `src.views.history_view` does not exist.

- [ ] **Step 3: Implement the history view**

Create `src/views/history_view.py`:

```python
from datetime import datetime
from typing import Any

import streamlit as st


def make_snippet(text: str, limit: int = 200) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "…"


def format_confidence(confidence: float) -> str:
    return f"{confidence * 100:.1f}%"


class HistoryView:
    @staticmethod
    def render(
        items: list[dict[str, Any]], offset: int, has_more: bool
    ) -> int | None:
        st.subheader("Analysis history")
        if not items:
            st.info("No analysis history yet.")
        for item in items:
            created = datetime.fromisoformat(
                item["created_at"].replace("Z", "+00:00")
            )
            heading = (
                f"{created:%Y-%m-%d %H:%M} — {item['prediction_label']} "
                f"({format_confidence(float(item['confidence']))})"
            )
            with st.expander(heading):
                st.caption(
                    f"Source: {item['input_source']} · "
                    f"Preview: {make_snippet(item['input_text'])}"
                )
                st.text_area(
                    "Analyzed text",
                    value=item["input_text"],
                    height=220,
                    disabled=True,
                    key=f"history_text_{item['id']}",
                )
                flags = item.get("red_flags") or []
                if flags:
                    st.markdown("**Red flags**")
                    for flag in flags:
                        st.write(f"- {flag}")
                else:
                    st.caption("No explicit heuristic red flags.")

        previous_col, next_col = st.columns(2)
        with previous_col:
            if st.button(
                "Previous",
                disabled=offset == 0,
                use_container_width=True,
            ):
                return max(0, offset - 20)
        with next_col:
            if st.button(
                "Next", disabled=not has_more, use_container_width=True
            ):
                return offset + 20
        return None

    @staticmethod
    def render_error(message: str) -> None:
        st.error(message)
```

- [ ] **Step 4: Change `MainView` to return source and render sidebar**

In `src/views/main_view.py`, import `AuthenticatedUser`, add this method to `MainView`, set `input_source`, and update both return statements:

```python
from src.models.auth_service import AuthenticatedUser


    @staticmethod
    def render_sidebar(user: AuthenticatedUser) -> tuple[str, bool]:
        with st.sidebar:
            st.write(f"Signed in as **{user.full_name}**")
            st.caption(user.email)
            page = st.radio("Navigation", ["Analyze", "History"])
            logout_clicked = st.button("Logout", use_container_width=True)
        return page, logout_clicked
```

At the start of `render_input_section`, use:

```python
        input_source = "text" if input_mode == "Paste Text" else "image"
```

Replace the oversized-image return with:

```python
                    return "", input_source, is_invalid
```

Replace the final return with:

```python
        return text, input_source, is_invalid
```

- [ ] **Step 5: Run formatting tests and compile the modified views**

```powershell
.\venv\Scripts\python.exe -m unittest tests.views.test_history_view -v
.\venv\Scripts\python.exe -m compileall -q src/views/main_view.py src/views/history_view.py
```

Expected: 2 tests PASS and compileall exits 0.

- [ ] **Step 6: Commit view contracts**

```powershell
git add src/views/main_view.py src/views/history_view.py tests/views/__init__.py tests/views/test_history_view.py
git commit -S -m "feat: add analysis history navigation"
```

---

### Task 7: Controller Auth Gate, Callback, Analysis Persistence, and Logout

**Files:**
- Modify: `src/controllers/app_controller.py:1-77`
- Create: `tests/controllers/__init__.py`
- Create: `tests/controllers/test_app_controller.py`

**Interfaces:**
- Consumes: all interfaces produced by Tasks 1-6.
- Produces: mandatory auth gate, token-hash callback handling, auth action dispatch, guarded classifier loading, paginated history navigation, and local-first logout cleanup.

- [ ] **Step 1: Write failing controller gate tests**

Create empty `tests/controllers/__init__.py`. Create `tests/controllers/test_app_controller.py`:

```python
import unittest
from unittest.mock import Mock, patch

from src.controllers.app_controller import AppController
from src.models.auth_service import AuthSession, AuthenticatedUser
from src.models.history_repository import HistoryError, HistoryPage
from src.models.session_store import save_auth_session


def session():
    return AuthSession(
        user=AuthenticatedUser(
            id="user-a", email="person@example.com", full_name="Person A"
        ),
        access_token="access-a",
        refresh_token="refresh-a",
    )


class AppControllerTests(unittest.TestCase):
    def build_controller(self, state=None):
        self.view = Mock()
        self.view.render_sidebar.return_value = ("Analyze", False)
        self.view.render_input_section.return_value = ("", "text", False)
        self.auth_view = Mock()
        self.auth_view.render_auth_page.return_value = None
        self.history_view = Mock()
        self.auth_service = Mock()
        self.history_repository = Mock()
        self.classifier_loader = Mock()
        self.query_params = {}
        return AppController(
            view=self.view,
            auth_view=self.auth_view,
            history_view=self.history_view,
            auth_service=self.auth_service,
            history_repository=self.history_repository,
            classifier_loader=self.classifier_loader,
            state=state if state is not None else {},
            query_params=self.query_params,
        )

    def test_anonymous_user_never_loads_classifier(self):
        controller = self.build_controller()
        controller.run()
        self.auth_view.render_auth_page.assert_called_once()
        self.classifier_loader.assert_not_called()

    def test_authenticated_user_loads_classifier_on_analyze(self):
        state = {}
        save_auth_session(state, session())
        controller = self.build_controller(state)
        self.auth_service.restore_session.return_value = session()
        self.classifier_loader.return_value = Mock(meta={})

        controller.run()

        self.classifier_loader.assert_called_once()

    @patch("src.controllers.app_controller.st")
    @patch("src.controllers.app_controller.time.sleep")
    @patch("src.controllers.app_controller.clean_text", return_value="cleaned")
    @patch("src.controllers.app_controller.check_red_flags", return_value=[])
    def test_history_failure_keeps_rendered_result(
        self, _flags, _clean, _sleep, streamlit
    ):
        state = {}
        save_auth_session(state, session())
        controller = self.build_controller(state)
        self.auth_service.restore_session.return_value = session()
        classifier = Mock(meta={})
        classifier.classify_text.return_value = ("Legitimate Job", 0.9)
        self.classifier_loader.return_value = classifier
        streamlit.status.return_value.__enter__.return_value = Mock()
        self.view.render_input_section.return_value = (
            "Job description", "text", False
        )
        self.view.render_result_section.side_effect = (
            lambda is_disabled, on_analyze: on_analyze()
        )
        self.history_repository.create.side_effect = HistoryError("save failed")

        controller.run()

        self.view.render_classification_result.assert_called_once()
        self.view.render_warning.assert_called_with(
            "Analysis history could not be saved."
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run controller tests to verify they fail**

```powershell
.\venv\Scripts\python.exe -m unittest tests.controllers.test_app_controller -v
```

Expected: FAIL because the current `AppController` does not accept injected dependencies or enforce auth.

- [ ] **Step 3: Replace `AppController` with the integrated orchestration**

Replace `src/controllers/app_controller.py` with:

```python
import time
from collections.abc import MutableMapping
from typing import Any, Callable

import streamlit as st

from src.models.auth_service import (
    AuthError,
    AuthService,
    AuthSession,
    ValidationError,
)
from src.models.classifier import ScamClassifier
from src.models.heuristics import check_red_flags
from src.models.history_repository import (
    AnalysisHistoryCreate,
    HistoryError,
    HistoryRepository,
)
from src.models.ocr_engine import extract_text_from_image
from src.models.preprocessor import clean_text
from src.models.session_store import (
    clear_auth_state,
    is_recovery_mode,
    load_auth_tokens,
    mark_recovery_mode,
    save_auth_session,
)
from src.models.supabase_client import (
    SupabaseConfigError,
    create_session_client,
    load_supabase_settings,
)
from src.views.auth_view import AuthAction, AuthView
from src.views.history_view import HistoryView
from src.views.main_view import MainView


@st.cache_resource
def get_classifier():
    classifier = ScamClassifier()
    try:
        classifier.load_model()
        return classifier
    except Exception as exc:
        return str(exc)


class AppController:
    def __init__(
        self,
        view: MainView | None = None,
        auth_view: AuthView | None = None,
        history_view: HistoryView | None = None,
        auth_service: AuthService | None = None,
        history_repository: HistoryRepository | None = None,
        classifier_loader: Callable[[], Any] = get_classifier,
        state: MutableMapping[str, Any] | None = None,
        query_params: MutableMapping[str, Any] | None = None,
    ):
        self.view = view or MainView()
        self.auth_view = auth_view or AuthView()
        self.history_view = history_view or HistoryView()
        self.state = state if state is not None else st.session_state
        self.query_params = (
            query_params if query_params is not None else st.query_params
        )
        self.classifier_loader = classifier_loader
        self.config_error: str | None = None
        self.view.setup_page()

        if auth_service is not None and history_repository is not None:
            self.auth_service = auth_service
            self.history_repository = history_repository
            return

        try:
            settings = load_supabase_settings(st.secrets)
            client = create_session_client(settings)
            self.auth_service = AuthService(client, settings.app_url)
            self.history_repository = HistoryRepository(client)
        except SupabaseConfigError as exc:
            self.config_error = str(exc)
            self.auth_service = None
            self.history_repository = None

    def run(self) -> None:
        self.view.render_header()
        if self.config_error:
            self.view.render_error(self.config_error)
            return

        callback_session = self._consume_auth_callback()
        if callback_session is not None:
            save_auth_session(self.state, callback_session)

        if is_recovery_mode(self.state):
            self._run_recovery_form()
            return

        current_session = self._restore_session()
        if current_session is None:
            action = self.auth_view.render_auth_page()
            if action is not None:
                self._handle_auth_action(action)
            return

        page, logout_clicked = self.view.render_sidebar(current_session.user)
        if logout_clicked:
            self._logout()
            return
        if page == "History":
            self._run_history(current_session)
        else:
            self._run_analysis(current_session)

    def _consume_auth_callback(self) -> AuthSession | None:
        token_hash = self.query_params.get("token_hash")
        otp_type = self.query_params.get("type")
        if not token_hash and not otp_type:
            return None
        self.query_params.clear()
        try:
            session = self.auth_service.verify_token(
                str(token_hash or ""), str(otp_type or "")
            )
        except (AuthError, ValidationError) as exc:
            self.auth_view.render_error(str(exc))
            return None
        if otp_type == "recovery":
            mark_recovery_mode(self.state, True)
        else:
            self.auth_view.render_success("Email verified successfully.")
        return session

    def _restore_session(self) -> AuthSession | None:
        tokens = load_auth_tokens(self.state)
        if tokens is None:
            return None
        try:
            session = self.auth_service.restore_session(*tokens)
        except AuthError:
            clear_auth_state(self.state)
            return None
        save_auth_session(self.state, session)
        return session

    def _handle_auth_action(self, action: AuthAction) -> None:
        try:
            if action.kind == "login":
                session = self.auth_service.sign_in(
                    action.payload["email"], action.payload["password"]
                )
                save_auth_session(self.state, session)
                self.state["history_offset"] = 0
                st.rerun()
            elif action.kind == "signup":
                self.auth_service.sign_up(
                    action.payload["full_name"],
                    action.payload["email"],
                    action.payload["password"],
                    action.payload["confirmation"],
                )
                self.auth_view.render_success(
                    "Check your email to verify the new account."
                )
            elif action.kind == "forgot_password":
                try:
                    self.auth_service.request_password_reset(
                        action.payload["email"]
                    )
                except AuthError:
                    pass
                self.auth_view.render_success(
                    "If an account exists for that email, a reset message will be sent."
                )
        except (AuthError, ValidationError) as exc:
            self.auth_view.render_error(str(exc))

    def _run_recovery_form(self) -> None:
        action = self.auth_view.render_recovery_form()
        if action is None:
            return
        try:
            self.auth_service.update_password(
                action.payload["password"], action.payload["confirmation"]
            )
        except (AuthError, ValidationError) as exc:
            self.auth_view.render_error(str(exc))
            return
        try:
            self.auth_service.sign_out()
        except Exception:
            pass
        clear_auth_state(self.state)
        self.state.pop("history_offset", None)
        self.auth_view.render_success(
            "Password updated. Sign in with your new password."
        )

    def _logout(self) -> None:
        try:
            self.auth_service.sign_out()
        except Exception:
            pass
        finally:
            clear_auth_state(self.state)
            self.state.pop("history_offset", None)
        st.rerun()

    def _run_history(self, session: AuthSession) -> None:
        offset = int(self.state.get("history_offset", 0))
        try:
            page = self.history_repository.list_page(
                session.user.id, offset=offset
            )
        except HistoryError as exc:
            self.history_view.render_error(str(exc))
            return
        next_offset = self.history_view.render(
            page.items, offset, page.has_more
        )
        if next_offset is not None:
            self.state["history_offset"] = next_offset
            st.rerun()

    def _run_analysis(self, session: AuthSession) -> None:
        classifier_or_error = self.classifier_loader()
        if isinstance(classifier_or_error, str):
            self.view.render_error(
                "Failed to load model from `./best_model/`. "
                "Run the research notebook first to export a model.\n\n"
                f"Error: {classifier_or_error}"
            )
            return
        classifier = classifier_or_error
        self.view.render_model_info(classifier.meta)

        def handle_image_upload(image):
            with st.spinner("Extracting text with OCR..."):
                try:
                    return extract_text_from_image(image)
                except Exception as exc:
                    self.view.render_error(str(exc))
                    return ""

        text, input_source, is_invalid = self.view.render_input_section(
            on_image_uploaded=handle_image_upload
        )

        def handle_analyze():
            if is_invalid:
                return
            if not text.strip():
                self.view.render_warning(
                    "Please provide a job description to analyze."
                )
                return
            with st.status(
                "Analyzing job description...", expanded=True
            ) as status:
                st.write("⏳ Reading and parsing text...")
                time.sleep(0.5)
                st.write("⏳ Cleaning HTML tags and URLs...")
                cleaned_text = clean_text(text)
                time.sleep(0.5)
                st.write("⏳ Tokenizing input for Transformer model...")
                time.sleep(0.7)
                st.write("⏳ Running sequence classification...")
                label, confidence = classifier.classify_text(cleaned_text)
                time.sleep(1.0)
                st.write("⏳ Extracting heuristics & red flags...")
                red_flags = check_red_flags(text)
                time.sleep(0.5)
                status.update(
                    label="✅ Analysis Complete!",
                    state="complete",
                    expanded=False,
                )

            self.view.render_classification_result(
                label, confidence, red_flags
            )
            try:
                self.history_repository.create(
                    AnalysisHistoryCreate(
                        user_id=session.user.id,
                        input_text=text,
                        input_source=input_source,
                        prediction_label=label,
                        confidence=confidence,
                        red_flags=red_flags,
                    )
                )
            except HistoryError:
                self.view.render_warning(
                    "Analysis history could not be saved."
                )

        self.view.render_result_section(
            is_disabled=is_invalid, on_analyze=handle_analyze
        )
```

- [ ] **Step 4: Run focused controller tests**

```powershell
.\venv\Scripts\python.exe -m unittest tests.controllers.test_app_controller -v
```

Expected: 3 tests PASS without sleeping or making network requests.

- [ ] **Step 5: Run the complete unit suite**

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Expected: all tests PASS with no network access.

- [ ] **Step 6: Commit controller integration**

```powershell
git add src/controllers/app_controller.py tests/controllers/__init__.py tests/controllers/test_app_controller.py
git commit -S -m "feat: require login and persist analyses"
```

---

### Task 8: Supabase Setup Documentation and End-to-End Verification

**Files:**
- Create: `docs/supabase-setup.md`
- Modify: `README.md:104-181`
- Modify: `README.id.md:104-181`

**Interfaces:**
- Consumes: all completed code and migration artifacts.
- Produces: reproducible project setup, email callbacks compatible with Streamlit, explicit demo/production boundaries, and evidence that acceptance criteria pass.

- [ ] **Step 1: Write exact Supabase setup instructions**

Create `docs/supabase-setup.md` with these required sections and exact templates:

````markdown
# Supabase Setup

## 1. Create or select the project

Use a development Supabase project while testing. Copy its Project URL and
active publishable key; never copy a secret or service-role key into the app.

## 2. Apply the migration

Review `supabase/migrations/20260712150000_create_analysis_history.sql`, apply
it first to the development project, then run Supabase Security and Performance
Advisors. Apply the same committed migration to production only after both the
RLS verification script and UI smoke test pass.

## 3. Configure application URLs

Set Site URL to the deployed application origin. Add
`http://localhost:8501` for local development and add only trusted deployed
origins to the redirect allowlist.

## 4. Configure Confirm Sign Up template

```html
<h2>Confirm your account</h2>
<p><a href="{{ .SiteURL }}?token_hash={{ .TokenHash }}&type=email">Confirm email</a></p>
```

## 5. Configure Reset Password template

```html
<h2>Reset your password</h2>
<p><a href="{{ .SiteURL }}?token_hash={{ .TokenHash }}&type=recovery">Choose a new password</a></p>
<p>If you did not request this change, ignore this email.</p>
```

Token-hash query parameters are consumed once and cleared immediately by the
Streamlit app. Access and refresh tokens are never placed in the URL.

## 6. Configure local secrets

Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml` and replace
the deliberately fake values. The real file is git-ignored.

## 7. Thesis-demo email limitation

Supabase built-in SMTP is restricted to project-team addresses, rate-limited,
best-effort, and not suitable for public use. Add the thesis-demo email as a
team member and test sign-up/recovery only with that authorized address.

## 8. Public deployment prerequisites

Before public launch, configure custom SMTP, SPF, DKIM, DMARC, CAPTCHA, trusted
redirect allowlists, appropriate auth rate limits, backups, monitoring, and a
production smoke test. None of these steps requires a service-role key in the
Streamlit application.
````

- [ ] **Step 2: Update both READMEs**

In `README.md`, extend the Architecture list with:

```markdown
- **Authentication** (`src/models/auth_service.py`): Supabase Auth handles mandatory email/password access, email verification, password recovery, session refresh, and logout.
- **History** (`src/models/history_repository.py`): Supabase Postgres stores each user's analyses behind ownership-based RLS policies.
```

Insert this section after the model-weights setup and rename the existing run heading to `### 5. Run the web app`:

```markdown
### 4. Configure Supabase

Create or select a Supabase project, apply the committed migration, and copy
`.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`. Replace its fake
values with the project URL, active publishable key, and application URL. Follow
[the Supabase setup guide](docs/supabase-setup.md) for RLS and email templates.

The built-in Supabase SMTP service is suitable only for thesis demos and sends
only to project-team addresses. Configure custom SMTP before public deployment.
```

Append these exact feature and stack bullets in their existing sections:

```markdown
- **Mandatory authentication:** email verification, login, password recovery, session refresh, and logout through Supabase Auth
- **Per-user history:** authenticated users can review their own analysis details; RLS prevents cross-user access

- **Authentication and database:** Supabase Auth, Supabase Postgres, PostgREST, PostgreSQL RLS
```

In `README.id.md`, extend the Arsitektur list with:

```markdown
- **Autentikasi** (`src/models/auth_service.py`): Supabase Auth menangani akses wajib dengan email/password, verifikasi email, pemulihan password, refresh sesi, dan logout.
- **Riwayat** (`src/models/history_repository.py`): Supabase Postgres menyimpan analisis setiap pengguna di balik policy RLS berbasis kepemilikan.
```

Insert this section after setup model weights and rename the existing run heading to `### 5. Jalankan aplikasi web`:

```markdown
### 4. Konfigurasi Supabase

Buat atau pilih project Supabase, terapkan migration yang sudah di-commit, lalu
salin `.streamlit/secrets.example.toml` menjadi `.streamlit/secrets.toml`. Ganti
nilai palsunya dengan URL project, publishable key aktif, dan URL aplikasi. Ikuti
[panduan setup Supabase](docs/supabase-setup.md) untuk RLS dan template email.

SMTP bawaan Supabase hanya sesuai untuk demo skripsi dan hanya mengirim ke alamat
anggota tim project. Konfigurasikan custom SMTP sebelum deployment publik.
```

Append these exact feature and stack bullets in their existing sections:

```markdown
- **Autentikasi wajib:** verifikasi email, login, pemulihan password, refresh sesi, dan logout melalui Supabase Auth
- **Riwayat per pengguna:** pengguna terautentikasi dapat meninjau detail analisisnya sendiri; RLS mencegah akses lintas pengguna

- **Autentikasi dan database:** Supabase Auth, Supabase Postgres, PostgREST, PostgreSQL RLS
```

Keep the existing dependency installation, Tesseract, model download, and `streamlit run app.py` instructions unchanged.

- [ ] **Step 3: Apply the schema only to the confirmed development project**

Before changing remote state, identify the exact Supabase project and obtain user approval. Apply the committed migration to that project, run `supabase/tests/analysis_history_rls.sql`, then run both Supabase advisors.

Expected:

- migration is recorded exactly once;
- RLS script finishes with `ROLLBACK` and no raised exception;
- Security Advisor reports no missing-RLS or unsafe-policy finding for `analysis_history`;
- Performance Advisor recognizes `analysis_history_user_created_idx` and reports no actionable issue for this table.

- [ ] **Step 4: Run static and unit verification**

```powershell
.\venv\Scripts\python.exe -m compileall -q app.py src tests
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\venv\Scripts\python.exe -m pip check
git diff --check
git status --short
```

Expected: compileall exits 0, all tests PASS, pip reports `No broken requirements found`, diff check is empty, and status lists only the intended documentation changes before commit.

- [ ] **Step 5: Run the authenticated Streamlit smoke test**

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Verify in order:

1. Missing secrets show only secret names and never values.
2. Anonymous users see auth forms and the classifier is not loaded.
3. Sign up validates name/email/password and sends confirmation to the authorized team email.
4. Confirmation consumes and clears `token_hash`.
5. Login unlocks Analyze and History.
6. Text and image analysis still enforce existing limits.
7. A completed result creates exactly one owned history row.
8. History shows 20 rows per request, details, empty state, and navigation.
9. A forced database insert failure leaves the inference result visible with a warning.
10. Recovery consumes and clears its token, updates the password, signs out, and permits login with the new password.
11. Logout clears local state and returns to auth forms.
12. A second user cannot see the first user's history.

- [ ] **Step 6: Commit docs and verification artifacts**

```powershell
git add docs/supabase-setup.md README.md README.id.md
git commit -S -m "docs: add Supabase deployment setup"
```

- [ ] **Step 7: Verify signed commit history and clean tree**

```powershell
git log --show-signature -8 --oneline
git status --short
```

Expected: every new implementation commit has a good GPG signature and the working tree is clean.
