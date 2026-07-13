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

`RedirectTo` comes from `APP_URL`; that URL must be present in the redirect
allowlist.

For the deployed application, also add this exact Redirect URL:

```text
https://job-scam.streamlit.app/?verified=true
```

## 4. Keep the default Confirm signup template

Do not customize the Confirm signup link. Keep Supabase's default template,
which uses `{{ .ConfirmationURL }}`. `AuthService.sign_up()` supplies this
production redirect:

```text
https://job-scam.streamlit.app/?verified=true
```

After Supabase verifies the account, Streamlit consumes `verified=true`, shows
`Your email has been successfully verified. You can now log in.` once, and
leaves the user logged out for manual login.

## 5. Configure Reset Password template

```html
<h2>Reset your password</h2>
<p><a href="{{ .RedirectTo }}?token_hash={{ .TokenHash }}&type=recovery">Choose a new password</a></p>
<p>If you did not request this change, ignore this email.</p>
```

Recovery token-hash query parameters are consumed once and cleared immediately
by the Streamlit app. Access and refresh tokens are never placed in the URL.

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
