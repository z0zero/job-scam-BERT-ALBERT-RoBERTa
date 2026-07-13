"""
Job Scam Detection - Streamlit Web Application (MVC Architecture)

Entry point for the application. All logic is delegated to the AppController.
"""

import streamlit as st

from src.controllers.app_controller import AppController
from src.controllers.email_verification_notice import (
    consume_email_verification_notice,
)


EMAIL_VERIFICATION_SUCCESS_MESSAGE = (
    "Your email has been successfully verified. You can now log in."
)


if __name__ == "__main__":
    if consume_email_verification_notice(st.query_params):
        st.success(EMAIL_VERIFICATION_SUCCESS_MESSAGE)

    controller = AppController()
    controller.run()
