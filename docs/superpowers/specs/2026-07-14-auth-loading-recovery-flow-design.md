# Authentication, Model Loading, and Password Recovery Fix Design

## Goal

Resolve three production issues in the Streamlit deployment:

1. The previous login form appears duplicated or remains visible after login.
2. The Analyze page looks stuck for several minutes during the first model load.
3. A password-reset email opens the normal login form instead of the change-password form.

## Confirmed root causes

### Stale or duplicated login form

After a successful login, the controller saves the Supabase session and calls `st.rerun()`. On the next run, the authenticated branch immediately starts loading the classifier. During that long cold start, the browser can continue showing elements from the previous run, including the login form, while the sidebar from the authenticated run is already visible. This creates the appearance of duplicated forms.

### Analyze appears stuck

`get_classifier()` is wrapped in `st.cache_resource`, but the first run still downloads or restores the Hugging Face artifacts, creates the tokenizer and model, loads metadata, and moves the model to the selected device. On Streamlit Community Cloud this can take several minutes. The built-in cache spinner only shows `Running get_classifier()`, which does not explain that the application is still working.

### Password recovery opens login

The received password-reset link redirects to a URL fragment such as:

```text
https://job-scam.streamlit.app/#access_token=...&type=recovery
```

URL fragments are handled only by the browser and are not available through `st.query_params`. The existing controller expects a server-visible callback:

```text
?token_hash=...&type=recovery
```

Because the controller never receives the fragment values, recovery mode is not activated and the normal login form is rendered.

## Recommended architecture

### 1. Stable authenticated transition

Keep the current controller and session model, but add an explicit loading state between authentication and the Analyze page.

After login succeeds:

1. Save the Supabase session.
2. Set a session-state marker such as `model_loading_pending = True`.
3. Trigger `st.rerun()`.
4. On the authenticated run, render the sidebar and a dedicated model-loading panel before calling the classifier loader.
5. Remove the marker after the classifier is ready or after a load error is displayed.

The loading panel replaces the former auth body with clear production copy:

```text
Loading AI model...
The first load may take several minutes. Please keep this tab open.
```

The application must never render the auth form and authenticated page in the same controller path.

### 2. Explicit classifier loading UI

Change the classifier cache decorator to:

```python
@st.cache_resource(show_spinner=False)
```

The controller owns the user-facing status instead of exposing Streamlit's generic `Running get_classifier()` message.

During model initialization, show a status or spinner with the exact message:

```text
Loading AI model...
The first load may take several minutes. Please keep this tab open.
```

When loading succeeds, replace the loading panel with the normal Analyze interface.

When loading fails:

- show a safe error containing the underlying load detail already returned by `get_classifier()`
- provide a `Retry loading model` button
- clear the cached classifier before rerunning so the retry executes a fresh load attempt

Keep `st.cache_resource` so subsequent reruns on the same Streamlit instance reuse the model.

### 3. Remove artificial Analyze delays

Remove the five `time.sleep()` calls from the analysis progress flow. They are presentation-only delays and add several seconds after the expensive model inference has already completed.

Keep the progress messages, but allow each stage to advance immediately according to actual computation.

### 4. Server-readable password-recovery callback

Keep the current custom recovery flow based on `token_hash` and `verify_otp()`.

The Supabase **Reset password** template must use:

```html
<h2>Reset your password</h2>
<p>
  <a href="{{ .RedirectTo }}?token_hash={{ .TokenHash }}&type=recovery">
    Choose a new password
  </a>
</p>
<p>If you did not request this change, ignore this email.</p>
```

The application flow remains:

```text
User clicks reset link
        ↓
Streamlit receives ?token_hash=...&type=recovery
        ↓
AuthService.verify_recovery_token()
        ↓
Recovery session is stored
        ↓
Choose a new password form is rendered
        ↓
Password is updated and the user returns to login
```

The **Confirm signup** template remains the Supabase default template. Only the **Reset password** template uses the custom token-hash link.

### 5. Recovery callback handling

The controller continues to consume recovery parameters before attempting normal session restoration.

When the callback succeeds:

- clear the callback parameters from the URL
- save the returned Supabase session
- mark recovery mode
- render only `Choose a new password`

When the callback is invalid or expired:

- clear the callback parameters
- clear recovery-related local state
- show a safe recovery-link error
- render the normal auth page on the next interaction

## Components

### `src/controllers/app_controller.py`

Responsibilities added or adjusted:

- manage the explicit classifier loading panel
- provide retry behavior for classifier load failures
- remove artificial analysis sleeps
- preserve the existing recovery-first routing
- ensure auth and authenticated bodies are mutually exclusive

### `src/models/session_store.py`

Add focused helpers for the temporary model-loading marker if needed. The marker must not be stored inside the authentication token payload.

### `src/views/main_view.py`

Add small focused view methods for:

- model-loading status
- model-load failure with retry action

Do not redesign unrelated Analyze or History UI.

### `src/models/auth_service.py`

Keep `verify_recovery_token(token_hash)` and `verify_otp({"token_hash": ..., "type": "recovery"})` unchanged unless tests reveal an SDK signature mismatch.

### `docs/supabase-setup.md`

Clarify that:

- Confirm signup uses the default template
- Reset password uses the custom `TokenHash` template
- the generated recovery link must contain `?token_hash=` and not `#access_token=`

## Security considerations

- Never parse or log `access_token`, `refresh_token`, or recovery tokens in plaintext diagnostics.
- Clear recovery query parameters immediately after consuming them.
- A loading-state marker provides no authentication authority.
- Authentication continues to depend on the restored Supabase session.
- Model retry must clear only the classifier cache, not user authentication state.
- Generic forgot-password messaging remains unchanged to avoid account enumeration.

## Error handling

### Model load error

Display a clear error and a retry button. Do not fall back to rendering the login form for an authenticated user.

### Expired recovery token

Display:

```text
This password recovery link is invalid or has expired.
```

Then allow the user to request a new reset email.

### Supabase configuration error

Continue using the existing configuration error path before authentication or model loading begins.

## Tests

Add or update tests covering:

1. Successful login stores the session, marks the authenticated transition, and reruns.
2. An authenticated cold start renders model-loading UI and never renders the auth page.
3. A successful classifier load clears the temporary loading marker.
4. A classifier error renders a retry control without clearing authentication.
5. Retry clears the classifier cache and reruns.
6. Analyze no longer invokes artificial sleep calls.
7. A `token_hash&type=recovery` callback enters recovery mode and renders only the update-password form.
8. Invalid or expired recovery callbacks clear URL parameters and local recovery state.
9. Forgot-password messaging remains generic.
10. Signup confirmation behavior remains unchanged.

## Production verification

After deployment:

1. Login with an existing confirmed account.
2. Verify the login form disappears and is replaced by the model-loading message.
3. Wait for the first model load and confirm Analyze appears without refreshing.
4. Refresh and confirm later model loads are substantially faster while the Streamlit instance remains warm.
5. Run one text analysis and confirm there are no artificial multi-second pauses.
6. Request a password reset.
7. Confirm the email link contains `?token_hash=` and `type=recovery`.
8. Click the link and confirm `Choose a new password` appears instead of Login.
9. Update the password and log in using the new password.

## Out of scope

- Moving inference to a separate API service
- Replacing Streamlit Community Cloud
- Background model preloading before the first authenticated user
- Converting the full auth flow to PKCE
- Reading URL fragments with a custom JavaScript component
- Redesigning the Analyze, History, or authentication pages
