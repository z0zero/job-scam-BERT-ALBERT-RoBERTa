# Email Verification Success Message Design

## Goal

Show a clear English success message after a user clicks the default Supabase **Confirm email address** link and returns to the Streamlit application.

Message:

> Your email has been successfully verified. You can now log in.

The user remains logged out and signs in manually.

## Current behavior

Supabase's default confirmation template verifies the account on the Supabase `/verify` endpoint and redirects the browser to the `email_redirect_to` value supplied during sign-up. The current Streamlit controller deliberately does not process sign-up verification tokens, so the application has no reliable signal that the redirect followed a successful email confirmation.

## Recommended approach

Keep the default Supabase confirmation template unchanged. Add a dedicated, non-secret query parameter to the sign-up redirect URL sent by `AuthService`:

```text
https://job-scam.streamlit.app/?verified=true
```

Supabase includes that destination in its default `{{ .ConfirmationURL }}`. After Supabase verifies the token, it redirects the browser to the marked application URL. The marker carries no token, user ID, email address, or credential.

The Streamlit controller will:

1. Read `verified` from `st.query_params` before rendering the authentication page.
2. Accept only the exact value `true`.
3. Remove only that parameter immediately so refreshing the page does not repeat the message.
4. Render the success banner through `AuthView.render_success()`.
5. Continue to the normal login page without creating an application session.

## Components

### `AuthService`

For sign-up only, construct the redirect URL by appending `?verified=true` to `APP_URL` and pass it through `email_redirect_to`.

Example:

```python
"email_redirect_to": f"{self.app_url}/?verified=true"
```

Password recovery continues using the unmodified `APP_URL` and its existing custom recovery callback.

The exact marked URL must be included in the Supabase Redirect URLs allowlist:

```text
https://job-scam.streamlit.app/?verified=true
```

The default Confirm signup email template remains untouched.

### `AppController`

Add a small method such as `_consume_email_verification_notice()` that:

- returns immediately when `verified` is absent
- removes only the `verified` parameter rather than clearing unrelated parameters
- displays `Your email has been successfully verified. You can now log in.` when the value equals `true`
- ignores unsupported values

Call it after configuration validation and before authentication-page rendering.

### `AuthView`

Reuse the existing `render_success()` method. No new visual component is required.

## Data flow

```text
User submits sign-up form
        ↓
AuthService sends email_redirect_to=APP_URL/?verified=true
        ↓
Default Supabase email contains ConfirmationURL
        ↓
User clicks Confirm email address
        ↓
Supabase verifies the confirmation token
        ↓
Supabase redirects to APP_URL/?verified=true
        ↓
AppController consumes verified=true
        ↓
Success banner is shown once above the login form
        ↓
Query marker is removed and user logs in manually
```

## Security and failure handling

The `verified=true` marker is only a UI notice. It must never be treated as proof of authentication or authorization. Login still requires Supabase password authentication, and database access remains protected by the Supabase session and RLS.

If someone manually opens `/?verified=true`, they may see the same informational banner, but they receive no session and no additional access. This is acceptable because the message is not a security boundary.

Unknown values such as `verified=false` or `verified=abc` are removed or ignored. Existing password-recovery callback parameters remain untouched.

## Tests

Add tests covering:

1. Sign-up sends `email_redirect_to` with `/?verified=true`.
2. `verified=true` renders the exact success message.
3. The marker is removed after consumption.
4. The classifier is not loaded for an anonymous verification-return visit.
5. Unsupported marker values do not show a success message.
6. Recovery callback parameters continue to work and are not removed by verification-notice handling.

## Out of scope

- Automatic login after email confirmation
- Persisting the success message across browser sessions
- Custom confirmation-token verification in Streamlit
- Modifying the default Supabase Confirm signup template
- Redesigning the authentication page
- Changing password-recovery behavior
