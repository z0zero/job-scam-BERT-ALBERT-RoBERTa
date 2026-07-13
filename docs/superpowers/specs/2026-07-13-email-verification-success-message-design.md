# Email Verification Success Message Design

## Goal

Show a clear English success message after a user clicks the default Supabase **Confirm email address** link and returns to the Streamlit application.

Message:

> Your email has been successfully verified. You can now log in.

The user remains logged out and signs in manually.

## Current behavior

Supabase's default confirmation template verifies the account on the Supabase `/verify` endpoint and redirects the browser to `APP_URL`. The current Streamlit controller deliberately does not process sign-up verification tokens, so the application has no reliable signal that the redirect came from a successful email confirmation.

## Recommended approach

Use a dedicated, non-secret query parameter on the post-verification redirect:

```text
https://job-scam.streamlit.app/?verified=true
```

Configure the default Supabase confirmation template so the confirmation URL redirects to the application with this marker. The marker carries no token, user ID, email address, or credential.

The Streamlit controller will:

1. Read `verified` from `st.query_params` before rendering the authentication page.
2. Accept only the exact value `true`.
3. Remove the parameter immediately so refreshing the page does not repeat the message.
4. Render the success banner through `AuthView.render_success()`.
5. Continue to the normal login page without creating an application session.

## Components

### Supabase Confirm signup template

Use Supabase's confirmation endpoint and set the redirect destination to:

```html
<a href="{{ .ConfirmationURL }}&redirect_to={{ .SiteURL }}/?verified=true">
  Confirm email address
</a>
```

The exact separator must match the generated `ConfirmationURL`. Before applying this template, verify whether `ConfirmationURL` already contains query parameters. If it does, append with `&redirect_to=` as shown.

### `AppController`

Add a small method such as `_consume_email_verification_notice()` that:

- returns immediately when `verified` is absent
- clears only the `verified` parameter rather than unrelated query parameters
- displays `Your email has been successfully verified. You can now log in.` when the value equals `true`
- ignores unsupported values

Call it after configuration validation and before authentication-page rendering.

### `AuthView`

Reuse the existing `render_success()` method. No new visual component is required.

## Data flow

```text
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

Unknown values such as `verified=false` or `verified=abc` are silently removed or ignored. Existing password-recovery callback parameters remain untouched.

## Tests

Add controller tests covering:

1. `verified=true` renders the exact success message.
2. The marker is removed after consumption.
3. The classifier is not loaded for an anonymous verification-return visit.
4. Unsupported marker values do not show a success message.
5. Recovery callback parameters continue to work and are not removed by verification-notice handling.

## Out of scope

- Automatic login after email confirmation
- Persisting the success message across browser sessions
- Custom confirmation-token verification in Streamlit
- Redesigning the authentication page
- Changing password-recovery behavior
