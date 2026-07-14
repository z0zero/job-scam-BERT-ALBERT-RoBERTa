# Password Update Login Redirect Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After a successful password recovery update, rerun to the normal Login form and show `Password updated. Sign in with your new password.` exactly once.

**Architecture:** Store a one-time UI notice in Streamlit session state, separate from Supabase authentication state. A successful recovery update clears auth/recovery state, stores the notice, and calls `st.rerun()`; the next anonymous controller run pops and renders the notice before the Login form. Failed updates keep the recovery form and do not create a notice or rerun.

**Tech Stack:** Python 3.12+, Streamlit, Supabase Python SDK 2.31.0, `unittest`, `unittest.mock`.

## Global Constraints

- Use the exact success message: `Password updated. Sign in with your new password.`
- Do not automatically log the user in after password update.
- Do not store the notice in query parameters, Supabase, or the auth-token payload.
- The notice is UI-only and must never grant authentication authority.
- Keep the notice separate from `clear_auth_state()` so it survives the intentional auth-state clear before rerun.
- A failed password update must keep recovery mode active, preserve the recovery session, and avoid rerunning.
- Do not redesign the Login or recovery forms.

---

## File Map

- Modify `src/models/session_store.py`: add one-time authentication notice helpers.
- Modify `tests/models/test_session_store.py`: verify notice round trip, one-time consumption, invalid values, and survival across auth-state clearing.
- Modify `src/controllers/app_controller.py`: store the notice and rerun after successful password update; consume the notice before rendering the Login form.
- Modify `tests/controllers/test_app_controller.py`: verify redirect-to-login behavior, one-time notice rendering, and failure behavior.

---

### Task 1: Add one-time authentication notice state

**Files:**
- Modify: `src/models/session_store.py:9-88`
- Modify: `tests/models/test_session_store.py`

**Interfaces:**
- Produces: `AUTH_NOTICE_KEY = "auth_notice"`.
- Produces: `set_auth_notice(state: MutableMapping[str, Any], message: str) -> None`.
- Produces: `pop_auth_notice(state: MutableMapping[str, Any]) -> str | None`.
- Preserves: `clear_auth_state(state)` must not remove `AUTH_NOTICE_KEY`.

- [ ] **Step 1: Add failing tests for notice storage and one-time consumption**

Update the imports in `tests/models/test_session_store.py` to include:

```python
from src.models.session_store import (
    AUTH_NOTICE_KEY,
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
    pop_auth_notice,
    save_auth_session,
    set_auth_notice,
)
```

Add these tests inside `SessionStoreTests`:

```python
def test_auth_notice_round_trip_is_consumed_once(self):
    set_auth_notice(
        self.state,
        "Password updated. Sign in with your new password.",
    )

    self.assertEqual(
        pop_auth_notice(self.state),
        "Password updated. Sign in with your new password.",
    )
    self.assertIsNone(pop_auth_notice(self.state))
    self.assertNotIn(AUTH_NOTICE_KEY, self.state)


def test_empty_auth_notice_is_not_stored(self):
    set_auth_notice(self.state, "")

    self.assertNotIn(AUTH_NOTICE_KEY, self.state)
    self.assertIsNone(pop_auth_notice(self.state))


def test_invalid_auth_notice_value_is_discarded(self):
    for value in (None, 1, True, [], {}):
        with self.subTest(value=value):
            self.state[AUTH_NOTICE_KEY] = value
            self.assertIsNone(pop_auth_notice(self.state))
            self.assertNotIn(AUTH_NOTICE_KEY, self.state)


def test_clear_auth_state_preserves_pending_notice(self):
    save_auth_session(self.state, self.session)
    mark_recovery_mode(self.state, True)
    mark_model_loading_pending(self.state, True)
    set_auth_notice(
        self.state,
        "Password updated. Sign in with your new password.",
    )

    clear_auth_state(self.state)

    self.assertIsNone(load_auth_session(self.state))
    self.assertFalse(is_recovery_mode(self.state))
    self.assertFalse(is_model_loading_pending(self.state))
    self.assertEqual(
        pop_auth_notice(self.state),
        "Password updated. Sign in with your new password.",
    )
```

- [ ] **Step 2: Run the focused session-store tests and confirm failure**

Run:

```bash
python -m unittest tests.models.test_session_store -v
```

Expected: import errors or test failures because `AUTH_NOTICE_KEY`, `set_auth_notice()`, and `pop_auth_notice()` do not exist yet.

- [ ] **Step 3: Implement the notice helpers**

In `src/models/session_store.py`, add the constant beside the existing state keys:

```python
AUTH_NOTICE_KEY = "auth_notice"
```

Add these functions immediately before `clear_auth_state()`:

```python
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
```

Do not add `AUTH_NOTICE_KEY` to `clear_auth_state()`.

- [ ] **Step 4: Run the session-store tests**

Run:

```bash
python -m unittest tests.models.test_session_store -v
```

Expected: all tests in `tests.models.test_session_store` pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/models/session_store.py tests/models/test_session_store.py
git commit -m "feat: add one-time auth notice state"
```

---

### Task 2: Redirect successful recovery updates to Login with a one-time message

**Files:**
- Modify: `src/controllers/app_controller.py:23-30,119-124,229-249`
- Modify: `tests/controllers/test_app_controller.py:18-24,216-309`

**Interfaces:**
- Consumes: `set_auth_notice(state, message)` from Task 1.
- Consumes: `pop_auth_notice(state) -> str | None` from Task 1.
- Produces: anonymous runs render a pending notice before `AuthView.render_auth_page()`.
- Produces: successful `_run_recovery_form()` calls `st.rerun()` after clearing auth state and storing the notice.

- [ ] **Step 1: Import notice helpers in controller tests**

Extend the `src.models.session_store` test import to include:

```python
pop_auth_notice,
set_auth_notice,
```

- [ ] **Step 2: Replace the current recovery-update test with a failing redirect test**

Replace `test_recovery_rerun_restores_session_before_updating_password` with:

```python
@patch("src.controllers.app_controller.st")
def test_successful_password_update_stores_notice_and_reruns(
    self,
    streamlit,
):
    state = {}
    save_auth_session(state, session())
    state["supabase_recovery_mode"] = True
    controller = self.build_controller(state)
    self.auth_service.restore_session.return_value = session()
    self.auth_view.render_recovery_form.return_value = AuthAction(
        "update_password",
        {
            "password": "new-password",
            "confirmation": "new-password",
        },
    )

    controller.run()

    self.auth_service.update_password.assert_called_once_with(
        "new-password",
        "new-password",
    )
    self.auth_service.sign_out.assert_called_once_with()
    self.assertIsNone(load_auth_tokens(state))
    self.assertFalse(is_recovery_mode(state))
    self.assertEqual(
        pop_auth_notice(state),
        "Password updated. Sign in with your new password.",
    )
    self.auth_view.render_success.assert_not_called()
    streamlit.rerun.assert_called_once_with()
    self.classifier_loader.assert_not_called()
```

This test verifies that success is no longer rendered in the same recovery-form run.

- [ ] **Step 3: Add failing tests for Login rendering and one-time notice consumption**

Add:

```python
def test_anonymous_run_renders_pending_notice_before_login(self):
    state = {}
    set_auth_notice(
        state,
        "Password updated. Sign in with your new password.",
    )
    controller = self.build_controller(state)

    controller.run()

    self.auth_view.render_success.assert_called_once_with(
        "Password updated. Sign in with your new password."
    )
    self.auth_view.render_auth_page.assert_called_once_with()
    self.assertIsNone(pop_auth_notice(state))
    self.classifier_loader.assert_not_called()


def test_auth_notice_is_not_repeated_on_later_anonymous_run(self):
    state = {}
    set_auth_notice(
        state,
        "Password updated. Sign in with your new password.",
    )
    first_controller = self.build_controller(state)
    first_controller.run()

    second_controller = self.build_controller(state)
    second_controller.run()

    self.auth_view.render_success.assert_not_called()
    self.auth_view.render_auth_page.assert_called_once_with()
    self.assertIsNone(pop_auth_notice(state))
```

Because `build_controller()` resets mocks, the second assertion proves the notice was consumed by the first run.

- [ ] **Step 4: Add a failing test for password-update failure**

Add:

```python
@patch("src.controllers.app_controller.st")
def test_failed_password_update_keeps_recovery_mode_without_notice_or_rerun(
    self,
    streamlit,
):
    state = {}
    save_auth_session(state, session())
    state["supabase_recovery_mode"] = True
    controller = self.build_controller(state)
    self.auth_service.restore_session.return_value = session()
    self.auth_service.update_password.side_effect = AuthError(
        "Password could not be updated. Try again."
    )
    self.auth_view.render_recovery_form.return_value = AuthAction(
        "update_password",
        {
            "password": "new-password",
            "confirmation": "new-password",
        },
    )

    controller.run()

    self.assertTrue(is_recovery_mode(state))
    self.assertEqual(load_auth_tokens(state), ("access-a", "refresh-a"))
    self.assertIsNone(pop_auth_notice(state))
    self.auth_view.render_error.assert_called_once_with(
        "Password could not be updated. Try again."
    )
    self.auth_service.sign_out.assert_not_called()
    streamlit.rerun.assert_not_called()
```

- [ ] **Step 5: Run the focused controller tests and confirm failure**

Run:

```bash
python -m unittest tests.controllers.test_app_controller -v
```

Expected: the new success-flow tests fail because the controller still renders the success message in the recovery run and never calls `st.rerun()`; anonymous runs do not yet consume notices.

- [ ] **Step 6: Import the notice helpers in the controller**

Extend the `src.models.session_store` import in `src/controllers/app_controller.py` with:

```python
pop_auth_notice,
set_auth_notice,
```

Add a module-level constant below `MODEL_LOADING_MESSAGE`:

```python
PASSWORD_UPDATED_MESSAGE = (
    "Password updated. Sign in with your new password."
)
```

- [ ] **Step 7: Render a pending notice before the Login form**

Change the anonymous branch in `_run_content()` from:

```python
if current_session is None:
    action = self.auth_view.render_auth_page()
```

to:

```python
if current_session is None:
    notice = pop_auth_notice(self.state)
    if notice is not None:
        self.auth_view.render_success(notice)
    action = self.auth_view.render_auth_page()
```

The notice must be popped before rendering so it cannot repeat on later reruns.

- [ ] **Step 8: Store the notice and rerun after successful password update**

Replace the final part of `_run_recovery_form()`:

```python
clear_auth_state(self.state)
self.state.pop("history_offset", None)
self.auth_view.render_success(
    "Password updated. Sign in with your new password."
)
```

with:

```python
clear_auth_state(self.state)
self.state.pop("history_offset", None)
set_auth_notice(self.state, PASSWORD_UPDATED_MESSAGE)
st.rerun()
```

Do not change the existing early return when `update_password()` raises `AuthError` or `ValidationError`.

- [ ] **Step 9: Run the focused tests**

Run:

```bash
python -m unittest tests.models.test_session_store tests.controllers.test_app_controller -v
```

Expected: all session-store and controller tests pass.

- [ ] **Step 10: Run syntax and full test checks**

Run:

```bash
python -m compileall src tests
python -m unittest discover -s tests -v
```

Expected: syntax compilation succeeds. The full suite ends with `OK`; if environment-only imports prevent discovery, record the exact traceback and require the focused suites from Step 9 to pass before PR review.

- [ ] **Step 11: Commit Task 2**

```bash
git add src/controllers/app_controller.py tests/controllers/test_app_controller.py
git commit -m "fix: redirect password updates to login"
```

- [ ] **Step 12: Perform production smoke verification**

After deployment:

```text
1. Request a new password-reset email.
2. Open the recovery link.
3. Enter matching valid passwords and click Update password.
4. Confirm the recovery form disappears immediately after rerun.
5. Confirm the normal Login form appears.
6. Confirm the green notice reads: Password updated. Sign in with your new password.
7. Refresh the page and confirm the notice no longer appears.
8. Log in with the new password.
```

- [ ] **Step 13: Open a pull request**

Create a PR from `fix/password-update-login-redirect` to `main`:

```text
Title: fix: redirect successful password updates to login

Summary:
- store a one-time authentication notice in Streamlit session state
- rerun to the anonymous Login form after a successful password update
- show the password-updated message exactly once
- preserve recovery state when password update fails

Validation:
- python -m unittest tests.models.test_session_store -v
- python -m unittest tests.controllers.test_app_controller -v
- python -m compileall src tests
- python -m unittest discover -s tests -v
```
