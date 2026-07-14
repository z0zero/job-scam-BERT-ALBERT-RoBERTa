# Authentication, Model Loading, and Password Recovery Fix Design

## Goal

Resolve three production issues in the Streamlit deployment:

1. The previous login form appears duplicated or remains visible after login.
2. The Analyze page looks stuck for several minutes during the first model load.
3. A password-reset email opens the normal login form instead of the change-password form.

## Confirmed root