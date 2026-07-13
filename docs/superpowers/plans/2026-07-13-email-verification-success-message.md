# Email Verification Success Message Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show `Your email has been successfully verified. You can now log in.` exactly once when Supabase redirects a confirmed user back to the Streamlit login page.

**Architecture:** Keep Supabase's default Confirm signup template unchanged. `AuthService.sign_up()` supplies a marked `email_redirect_to` ending in `/?verified=true`, and `AppController` consumes only that query parameter, renders the existing success banner, removes the marker, and continues to the normal logged-out login page. Password recovery remains on its existing custom callback flow.

**Tech Stack:** Python 3.12+, Streamlit, Supabase Python SDK 2.31.0, `unittest`/`unittest.mock`.

## Global Constraints

- Keep the default Supabase Confirm signup email template unchanged.
- Display the exact English message: `Your email has been successfully verified. You can now log in.`
- Do not automatically sign the user in after confirmation.
- Treat `verified=true` only as a UI notice, never as authentication or authorization evidence.
- Remove only the `verified` query parameter; preserve password-recovery parameters and unrelated query parameters.
- Accept only the exact string value `true`; unsupported values must not display a success message.
- Keep password recovery behavior unchanged.
- Add `https://job-scam.streamlit.app/?verified=true` to the Supabase Redirect URLs allowlist before production verification.

---

## File Map

- Modify `src/models/auth_service.py`: send the marked sign-up redirect URL to Supabase.
- Modify `tests/models/test_auth_service.py`: lock down the marked redirect and ensure password recovery still uses the unmarked application URL.
- Modify `src/controllers/app_controller.py`: consume the verification marker and render the success notice once.
- Modify `tests/controllers/test_app_controller.py`: cover the success notice, invalid markers, parameter preservation, anonymous behavior, and existing recovery callback API.
- Modify `docs/supabase-setup.md`: document the default confirmation template and required marked redirect allowlist entry.

---

### Task 1: Mark the Supabase sign-up redirect

**Files:**
- Modify: `tests/models/test_auth_service.py:51-70`
- Modify: `src/models/auth_service.py:64-89`

**Interfaces:**
- Consumes: `AuthService.__init__(client: Client, app_url: str)` where `app_url` has no trailing slash.
- Produces: `AuthService._email_verification_redirect() -> str`, used only by `sign_up()`.
- Preserves: `request_password_reset()` continues to pass `self.app_url` unchanged.

- [ ] **Step 1: Update the sign-up test so it fails against the current implementation**

Replace `test_sign_up_uses_supabase_default_confirmation_redirect` with:

```python
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
```

Keep `test_reset_request_uses_exact_signature` unchanged so it continues asserting:

```python
{"redirect_to": "https://app.example.com"}
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
python -m unittest tests.models.test_auth_service.AuthServiceTests.test_sign_up_marks_default_confirmation_redirect -v
```

Expected: `FAIL` because the current call still sends `https://app.example.com` without `/?verified=true`.

- [ ] **Step 3: Add the minimal redirect helper and use it for sign-up only**

In `AuthService`, directly after `__init__`, add:

```python
def _email_verification_redirect(self) -> str:
    return f"{self.app_url}/?verified=true"
```

Then change the sign-up options from:

```python
"email_redirect_to": self.app_url,
```

to:

```python
"email_redirect_to": self._email_verification_redirect(),
```

Do not modify `request_password_reset()`.

- [ ] **Step 4: Run the focused model tests**

Run:

```bash
python -m unittest tests.models.test_auth_service -v
```

Expected: all tests in `tests.models.test_auth_service` pass.

- [ ] **Step 5: Commit the service change**

```bash
git add src/models/auth_service.py tests/models/test_auth_service.py
git commit -m "feat: mark email verification redirect"
```

---

### Task 2: Consume the verification marker and show the success banner once

**Files:**
- Modify: `tests/controllers/test_app_controller.py:26-119`
- Modify: `src/controllers/app_controller.py:89-146`

**Interfaces:**
- Consumes: mutable query-parameter mapping supporting `.get()` and key deletion.
- Produces: `AppController._consume_email_verification_notice() -> None`.
- Uses: `AuthView.render_success(message: str)`.
- Preserves: `AppController._consume_recovery_callback() -> AuthSession | None` and `AuthService.verify_recovery_token(token_hash: str)`.

- [ ] **Step 1: Add failing controller tests for a valid and invalid marker**

Add these tests after `test_anonymous_user_never_loads_classifier`:

```python
def test_verified_marker_shows_success_once_without_loading_classifier(self):
    params = {"verified": "true"}
    controller = self.build_controller(query_params=params)

    controller.run()

    self.auth_view.render_success.assert_called_once_with(
        "Your email has been successfully verified. You can now log in."
    )
    self.auth_view.render_auth_page.assert_called_once_with()
    self.assertEqual(params, {})
    self.classifier_loader.assert_not_called()


def test_unsupported_verified_marker_is_removed_without_success(self):
    params = {"verified": "false", "source": "email"}
    controller = self.build_controller(query_params=params)

    controller.run()

    self.auth_view.render_success.assert_not_called()
    self.assertEqual(params, {"source": "email"})
    self.auth_view.render_auth_page.assert_called_once_with()
    self.classifier_loader.assert_not_called()
```

- [ ] **Step 2: Update and extend the recovery callback test**

The current test still references the removed `verify_token()` API. Replace `test_callback_is_consumed_and_recovery_never_loads_classifier` with:

```python
def test_recovery_callback_survives_verification_notice_handling(self):
    state = {}
    params = {
        "verified": "true",
        "token_hash": "secret-token",
        "type": "recovery",
    }
    controller = self.build_controller(state, params)
    self.auth_service.verify_recovery_token.return_value = session()
    self.auth_view.render_recovery_form.return_value = None

    controller.run()

    self.auth_view.render_success.assert_called_once_with(
        "Your email has been successfully verified. You can now log in."
    )
    self.auth_service.verify_recovery_token.assert_called_once_with(
        "secret-token"
    )
    self.assertEqual(params, {})
    self.assertTrue(is_recovery_mode(state))
    self.classifier_loader.assert_not_called()
```

In `test_recovery_rerun_restores_session_before_updating_password`, replace:

```python
self.auth_service.verify_token.return_value = session()
```

with:

```python
self.auth_service.verify_recovery_token.return_value = session()
```

- [ ] **Step 3: Run the focused controller tests and confirm they fail**

Run:

```bash
python -m unittest tests.controllers.test_app_controller -v
```

Expected: the new verification-marker tests fail because `_consume_email_verification_notice()` does not exist, while the corrected recovery tests reveal any stale API references.

- [ ] **Step 4: Implement one-time notice consumption**

Add this module-level constant below the imports:

```python
EMAIL_VERIFICATION_SUCCESS_MESSAGE = (
    "Your email has been successfully verified. You can now log in."
)
```

In `run()`, after the configuration error guard and before `_consume_recovery_callback()`, add:

```python
self._consume_email_verification_notice()
```

Add this method immediately before `_consume_recovery_callback()`:

```python
def _consume_email_verification_notice(self) -> None:
    marker = self.query_params.get("verified")
    if marker is None:
        return

    try:
        del self.query_params["verified"]
    except KeyError:
        pass

    if str(marker) == "true":
        self.auth_view.render_success(
            EMAIL_VERIFICATION_SUCCESS_MESSAGE
        )
```

This method must not clear the whole query mapping and must not save an auth session.

- [ ] **Step 5: Run the focused controller tests**

Run:

```bash
python -m unittest tests.controllers.test_app_controller -v
```

Expected: all controller tests pass, including recovery callback tests.

- [ ] **Step 6: Commit the controller change**

```bash
git add src/controllers/app_controller.py tests/controllers/test_app_controller.py
git commit -m "feat: show email verification success notice"
```

---

### Task 3: Document Supabase production configuration and verify the complete change

**Files:**
- Modify: `docs/supabase-setup.md:17-42`
- Test: all files under `tests/`

**Interfaces:**
- Supabase Redirect URLs allowlist contains `https://job-scam.streamlit.app/?verified=true`.
- Supabase Confirm signup template remains the default template using `{{ .ConfirmationURL }}`.
- Password recovery keeps the repository's existing custom recovery template.

- [ ] **Step 1: Replace the Confirm signup documentation**

Replace the current custom Confirm signup template section with:

```markdown
## 4. Keep the default Confirm signup template

Do not customize the Confirm signup link. Keep Supabase's default template,
which uses `{{ .ConfirmationURL }}`. `AuthService.sign_up()` supplies this
production redirect:

```text
https://job-scam.streamlit.app/?verified=true
```

Add that exact URL to **Authentication → URL Configuration → Redirect URLs**.
After Supabase verifies the account, Streamlit consumes `verified=true`, shows
the success message once, and leaves the user logged out for manual login.
```

Keep the password-reset template section unchanged.

- [ ] **Step 2: Run all authentication and controller tests**

Run:

```bash
python -m unittest tests.models.test_auth_service tests.controllers.test_app_controller -v
```

Expected: all listed tests pass.

- [ ] **Step 3: Run the full test suite**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: the suite completes with `OK`. If unrelated environment-dependent model imports prevent discovery, record the exact traceback and still require the focused authentication and controller suites to pass before review.

- [ ] **Step 4: Perform a production smoke test**

In Supabase, add this exact Redirect URL:

```text
https://job-scam.streamlit.app/?verified=true
```

Then deploy the branch and test with a fresh email address:

```text
1. Submit Sign up once.
2. Open the confirmation email.
3. Click Confirm email address.
4. Confirm the browser returns to https://job-scam.streamlit.app/.
5. Confirm the green banner says: Your email has been successfully verified. You can now log in.
6. Refresh the page and confirm the banner does not appear again.
7. Log in manually and confirm analysis/history access still works.
8. Request password recovery and confirm its custom callback still opens the password form.
```

- [ ] **Step 5: Commit the documentation**

```bash
git add docs/supabase-setup.md
git commit -m "docs: explain verification success redirect"
```

- [ ] **Step 6: Open a pull request**

Create a PR from `feat/email-verification-success-message` to `main` with:

```text
Title: feat: show email verification success message

Summary:
- mark the default Supabase confirmation redirect with verified=true
- show a one-time success banner on the Streamlit login page
- preserve the existing password-recovery callback
- document the required Supabase redirect allowlist entry

Validation:
- python -m unittest tests.models.test_auth_service -v
- python -m unittest tests.controllers.test_app_controller -v
- python -m unittest discover -s tests -v
```
