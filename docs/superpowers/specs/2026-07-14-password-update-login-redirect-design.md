# Password Update Login Redirect Design

## Goal

After a successful password recovery update, immediately leave recovery mode, rerun the Streamlit app, show the normal Login form, and display this success notice exactly once:

```text
Password updated. Sign in with your new password.
```

## Confirmed root cause

`AppController._run_recovery_form()` currently updates the password, signs out, clears authentication state, and renders a success banner in the same script run. Because it does not call `st.rerun()`, the already-rendered `Choose a new password` form remains visible above the banner.

## Recommended architecture

Use a one-time authentication notice stored in Streamlit session state.

### Session-state helper

Add a dedicated key and two helpers in `src/models/session_store.py`:

```python
AUTH_NOTICE_KEY = "auth_notice"


def set_auth_notice(state, message: str) -> None:
    if message:
        state[AUTH_NOTICE_KEY] = message
    else:
        state.pop(AUTH_NOTICE_KEY, None)


def pop_auth_notice(state) -> str | None:
    value = state.pop(AUTH_NOTICE_KEY, None)
    return value if isinstance(value, str) and value else None
```

The notice is UI-only. It must not be stored inside the Supabase authentication payload and must not grant authentication authority.

### Successful recovery flow

After `AuthService.update_password()` succeeds:

1. Attempt Supabase sign-out as today.
2. Clear auth tokens, recovery mode, model-loading marker, and history offset.
3. Store the exact success notice with `set_auth_notice()`.
4. Call `st.rerun()`.
5. On the next anonymous run, consume the notice with `pop_auth_notice()`.
6. Render the success banner once, followed by the normal Login form.

### Notice rendering order

In the anonymous branch of `AppController._run_content()`:

1. Consume and render any authentication notice.
2. Render `AuthView.render_auth_page()`.

This guarantees the user sees the success message together with the Login form.

### Failure behavior

If password validation or Supabase update fails:

- keep recovery mode active
- keep the recovery session
- show the existing safe error
- do not set an authentication notice
- do not rerun
- keep the `Choose a new password` form visible

## Files

### `src/models/session_store.py`

- add `AUTH_NOTICE_KEY`
- add `set_auth_notice(state, message)`
- add `pop_auth_notice(state)`
- keep the notice separate from `clear_auth_state()` so the post-recovery message survives the deliberate auth-state clear before rerun

### `src/controllers/app_controller.py`

- import `set_auth_notice` and `pop_auth_notice`
- after a successful password update, save the notice and call `st.rerun()`
- in the anonymous branch, consume and render the notice before the Login form

### `tests/models/test_session_store.py`

Cover:

- notice round trip
- notice is removed when popped
- invalid or empty values are not returned
- clearing auth state does not delete a pending notice

### `tests/controllers/test_app_controller.py`

Cover:

- successful password update stores notice and reruns
- recovery form is not rendered on the next anonymous run
- Login form is rendered after rerun
- success notice appears exactly once
- a later rerun does not repeat the notice
- failed password update does not set notice or rerun

## Security considerations

- Do not place the message in query parameters.
- Do not persist it in Supabase.
- Do not include tokens or provider errors in the message.
- The notice must never be interpreted as proof that a user is authenticated.

## Out of scope

- Automatic login after password update
- Redirecting to Analyze immediately
- Changing the password-recovery email template
- Redesigning the Login or recovery forms
- Adding a general-purpose notification framework

## Production verification

1. Request a new password-reset email.
2. Open the recovery link.
3. Submit a valid new password.
4. Confirm the page reruns to the normal Login form.
5. Confirm the green notice says `Password updated. Sign in with your new password.`
6. Refresh the page and confirm the notice no longer appears.
7. Log in using the new password.
