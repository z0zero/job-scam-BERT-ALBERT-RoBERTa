# Authentication, Model Loading, and Password Recovery Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the stale/double login form during authenticated cold starts, explain long model initialization, provide safe retry behavior, and make password-recovery links open the change-password form.

**Architecture:** Render all dynamic main-page content inside one stable Streamlit placeholder so reruns replace the previous authentication body. Keep the classifier in `st.cache_resource`, suppress the generic cache spinner, and render an explicit loading panel before the blocking model load. Preserve the existing Supabase `token_hash` recovery flow and document the required custom Reset password template.

**Tech Stack:** Python 3.12, Streamlit, Supabase Python SDK, Hugging Face Transformers, `unittest`, `unittest.mock`.

## Global Constraints

- Keep authentication and authenticated content mutually exclusive on each controller run.
- Display `Loading AI model...` and `The first load may take several minutes. Please keep this tab open.` before classifier initialization.
- Keep `st.cache_resource`; set `show_spinner=False`.
- Retry must clear only the classifier cache and must preserve the authenticated Supabase session.
- Remove every artificial `time.sleep()` call from Analyze.
- Keep Confirm signup on the default Supabase template.
- Reset password must use `?token_hash={{ .TokenHash }}&type=recovery`, not `#access_token=...`.
- Never log or persist recovery tokens outside the existing one-time callback flow.
- Keep forgot-password responses generic to prevent account enumeration.

---

### Task 1: Add an authenticated model-loading transition marker

**Files:**
- Modify: `src/models/session_store.py`
- Modify: `tests/models/test_session_store.py`

**Interfaces:**
- Produce: `MODEL_LOADING_PENDING_KEY = "model_loading_pending"`
- Produce: `mark_model_loading_pending(state, enabled) -> None`
- Produce: `is_model_loading_pending(state) -> bool`
- Preserve: existing authentication and recovery-state helpers.

- [ ] **Step 1: Add failing tests**

```python
def test_model_loading_marker_is_explicit_and_removable(self):
    self.assertFalse(is_model_loading_pending(self.state))
    mark_model_loading_pending(self.state, True)
    self.assertTrue(is_model_loading_pending(self.state))
    mark_model_loading_pending(self.state, False)
    self.assertFalse(is_model_loading_pending(self.state))
```

Also extend the clear-state test to assert that `MODEL_LOADING_PENDING_KEY` is removed.

- [ ] **Step 2: Run the focused tests and confirm RED**

```bash
python -m unittest tests.models.test_session_store -v
```

Expected: failures because the loading-state constant and helpers do not exist.

- [ ] **Step 3: Implement the focused state helpers**

```python
MODEL_LOADING_PENDING_KEY = "model_loading_pending"


def mark_model_loading_pending(state, enabled: bool) -> None:
    if enabled is True:
        state[MODEL_LOADING_PENDING_KEY] = True
    else:
        state.pop(MODEL_LOADING_PENDING_KEY, None)


def is_model_loading_pending(state) -> bool:
    return state.get(MODEL_LOADING_PENDING_KEY) is True
```

Update `clear_auth_state()` so logout and invalid sessions remove the marker.

- [ ] **Step 4: Run tests and confirm GREEN**

```bash
python -m unittest tests.models.test_session_store -v
```

Expected: all session-store tests pass.

---

### Task 2: Stabilize the authenticated body and model-loading flow

**Files:**
- Modify: `src/views/main_view.py`
- Modify: `src/controllers/app_controller.py`
- Modify: `tests/controllers/test_app_controller.py`

**Interfaces:**
- Produce: `MainView.content_container()` returning a Streamlit container context.
- Produce: `MainView.render_model_loading(message)` returning a clearable placeholder.
- Produce: `MainView.render_model_load_error(detail) -> bool`.
- Extend: `AppController.__init__()` with `classifier_cache_clearer` for testable retry behavior.

- [ ] **Step 1: Add controller tests for the production symptoms**

Cover these behaviors:

```text
- login saves the session, marks model loading, and reruns
- an authenticated run never renders the auth page
- loading UI is emitted before the classifier loader runs
- load failure preserves auth and renders Retry loading model
- retry clears only the classifier cache and reruns
- History does not load the classifier
- Analyze contains no artificial sleep calls
```

Correct stale recovery tests to call:

```python
self.auth_service.verify_recovery_token("secret-token")
```

instead of the removed `verify_token()` API.

- [ ] **Step 2: Run the controller tests and confirm RED**

```bash
python -m unittest tests.controllers.test_app_controller -v
```

Expected: failures for missing view methods, missing transition state, and stale recovery API assertions.

- [ ] **Step 3: Add the stable dynamic body**

In `MainView`:

```python
@staticmethod
def content_container():
    return st.empty().container()
```

In `AppController.run()`:

```python
self.view.render_header()
with self.view.content_container():
    self._run_content()
```

Move the existing routing body into `_run_content()` so each rerun replaces the old login subtree before authenticated content is rendered.

- [ ] **Step 4: Add explicit classifier loading and retry**

Use:

```python
@st.cache_resource(show_spinner=False)
def get_classifier():
    ...
```

Before calling the loader, render:

```text
Loading AI model...
The first load may take several minutes. Please keep this tab open.
```

On failure, render the safe detail and a `Retry loading model` button. On retry:

```python
self.classifier_cache_clearer()
mark_model_loading_pending(self.state, True)
st.rerun()
```

Do not clear authentication state.

- [ ] **Step 5: Remove artificial Analyze delays**

Delete the `time` import and every `time.sleep()` call. Keep progress messages tied to actual parsing, preprocessing, inference, and heuristic work.

- [ ] **Step 6: Run focused tests and confirm GREEN**

```bash
python -m unittest tests.models.test_session_store tests.controllers.test_app_controller -v
```

Expected: all focused tests pass.

---

### Task 3: Lock down the recovery flow and production configuration

**Files:**
- Modify: `src/controllers/app_controller.py`
- Modify: `tests/controllers/test_app_controller.py`
- Modify: `docs/supabase-setup.md`

**Interfaces:**
- Consume: `?token_hash=<opaque-token>&type=recovery`
- Call: `AuthService.verify_recovery_token(token_hash)`
- Render: `AuthView.render_recovery_form()` only after a valid recovery session is stored.

- [ ] **Step 1: Test valid and invalid recovery callbacks**

Valid callback expectations:

```text
- query parameters are cleared
- verify_recovery_token receives the opaque token unchanged
- recovery mode is enabled
- Choose a new password form renders
- login and classifier do not render
```

Invalid callback expectations:

```text
- query parameters are cleared
- recovery mode is disabled
- safe expired-link error renders
- classifier does not load
```

- [ ] **Step 2: Normalize invalid recovery errors**

Display exactly:

```text
This password recovery link is invalid or has expired.
```

Do not expose Supabase provider internals.

- [ ] **Step 3: Document the required Supabase template**

Use this Reset password template:

```html
<h2>Reset your password</h2>
<p>
  <a href="{{ .RedirectTo }}?token_hash={{ .TokenHash }}&type=recovery">
    Choose a new password
  </a>
</p>
<p>If you did not request this change, ignore this email.</p>
```

Document that old messages keep their old fragment links and a new reset email must be requested after saving the template.

- [ ] **Step 4: Run verification**

```bash
python -m unittest tests.models.test_session_store tests.controllers.test_app_controller tests.models.test_auth_service -v
python -m unittest discover -s tests -v
```

Expected: focused authentication/controller tests pass. If the full suite is blocked by unavailable model-system dependencies, record the exact environment limitation and still require syntax compilation plus focused tests.

- [ ] **Step 5: Production smoke test**

```text
1. Login with a confirmed account.
2. Confirm the old login form is immediately replaced by the model-loading panel.
3. Wait for the cold start and confirm Analyze appears without manual refresh.
4. Run one analysis and confirm no presentation-only delays remain.
5. Request a new password-reset email after saving the custom template.
6. Confirm the URL contains ?token_hash= and type=recovery, not #access_token=.
7. Confirm Choose a new password renders, update the password, and log in again.
```
