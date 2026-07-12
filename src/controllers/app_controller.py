import time
from collections.abc import MutableMapping
from typing import Any, Callable

import streamlit as st

from src.models.auth_service import (
    AuthError,
    AuthService,
    AuthSession,
    ValidationError,
)
from src.models.classifier import ScamClassifier
from src.models.heuristics import check_red_flags
from src.models.history_repository import (
    AnalysisHistoryCreate,
    HistoryError,
    HistoryRepository,
)
from src.models.ocr_engine import extract_text_from_image
from src.models.preprocessor import clean_text
from src.models.session_store import (
    clear_auth_state,
    is_recovery_mode,
    load_auth_tokens,
    mark_recovery_mode,
    save_auth_session,
)
from src.models.supabase_client import (
    SupabaseConfigError,
    create_session_client,
    load_supabase_settings,
)
from src.views.auth_view import AuthAction, AuthView
from src.views.history_view import HistoryView
from src.views.main_view import MainView


@st.cache_resource
def get_classifier():
    classifier = ScamClassifier()
    try:
        classifier.load_model()
        return classifier
    except Exception as exc:
        return str(exc)


class AppController:
    def __init__(
        self,
        view: MainView | None = None,
        auth_view: AuthView | None = None,
        history_view: HistoryView | None = None,
        auth_service: AuthService | None = None,
        history_repository: HistoryRepository | None = None,
        classifier_loader: Callable[[], Any] = get_classifier,
        state: MutableMapping[str, Any] | None = None,
        query_params: MutableMapping[str, Any] | None = None,
    ):
        self.view = view or MainView()
        self.auth_view = auth_view or AuthView()
        self.history_view = history_view or HistoryView()
        self.state = state if state is not None else st.session_state
        self.query_params = (
            query_params if query_params is not None else st.query_params
        )
        self.classifier_loader = classifier_loader
        self.config_error: str | None = None
        self.view.setup_page()

        if auth_service is not None and history_repository is not None:
            self.auth_service = auth_service
            self.history_repository = history_repository
            return

        try:
            settings = load_supabase_settings(st.secrets)
            client = create_session_client(settings)
            self.auth_service = AuthService(client, settings.app_url)
            self.history_repository = HistoryRepository(client)
        except SupabaseConfigError as exc:
            self.config_error = str(exc)
            self.auth_service = None
            self.history_repository = None

    def run(self) -> None:
        self.view.render_header()
        if self.config_error:
            self.view.render_error(self.config_error)
            return

        callback_session = self._consume_auth_callback()
        if callback_session is not None:
            save_auth_session(self.state, callback_session)

        if is_recovery_mode(self.state):
            if not self._restore_recovery_session():
                return
            self._run_recovery_form()
            return

        current_session = self._restore_session()
        if current_session is None:
            action = self.auth_view.render_auth_page()
            if action is not None:
                self._handle_auth_action(action)
            return

        page, logout_clicked = self.view.render_sidebar(current_session.user)
        if logout_clicked:
            self._logout()
            return
        if page == "History":
            self._run_history(current_session)
        else:
            self._run_analysis(current_session)

    def _consume_auth_callback(self) -> AuthSession | None:
        token_hash = self.query_params.get("token_hash")
        otp_type = self.query_params.get("type")
        if not token_hash and not otp_type:
            return None
        self.query_params.clear()
        try:
            session = self.auth_service.verify_token(
                str(token_hash or ""), str(otp_type or "")
            )
        except (AuthError, ValidationError) as exc:
            self.auth_view.render_error(str(exc))
            return None
        if otp_type == "recovery":
            mark_recovery_mode(self.state, True)
        else:
            self.auth_view.render_success("Email verified successfully.")
        return session

    def _restore_session(self) -> AuthSession | None:
        tokens = load_auth_tokens(self.state)
        if tokens is None:
            return None
        try:
            session = self.auth_service.restore_session(*tokens)
        except (AuthError, ValidationError):
            clear_auth_state(self.state)
            return None
        save_auth_session(self.state, session)
        return session

    def _restore_recovery_session(self) -> bool:
        tokens = load_auth_tokens(self.state)
        if tokens is None:
            clear_auth_state(self.state)
            self.auth_view.render_error(
                "Recovery link or session has expired. Request a new password reset."
            )
            return False
        try:
            session = self.auth_service.restore_session(*tokens)
        except (AuthError, ValidationError):
            clear_auth_state(self.state)
            self.auth_view.render_error(
                "Recovery link or session has expired. Request a new password reset."
            )
            return False
        save_auth_session(self.state, session)
        return True

    def _handle_auth_action(self, action: AuthAction) -> None:
        try:
            if action.kind == "login":
                session = self.auth_service.sign_in(
                    action.payload["email"], action.payload["password"]
                )
                save_auth_session(self.state, session)
                self.state["history_offset"] = 0
                st.rerun()
            elif action.kind == "signup":
                self.auth_service.sign_up(
                    action.payload["full_name"],
                    action.payload["email"],
                    action.payload["password"],
                    action.payload["confirmation"],
                )
                self.auth_view.render_success(
                    "Check your email to verify the new account."
                )
            elif action.kind == "forgot_password":
                try:
                    self.auth_service.request_password_reset(
                        action.payload["email"]
                    )
                except AuthError:
                    pass
                self.auth_view.render_success(
                    "If an account exists for that email, a reset message will be sent."
                )
        except (AuthError, ValidationError) as exc:
            self.auth_view.render_error(str(exc))

    def _run_recovery_form(self) -> None:
        action = self.auth_view.render_recovery_form()
        if action is None:
            return
        try:
            self.auth_service.update_password(
                action.payload["password"], action.payload["confirmation"]
            )
        except (AuthError, ValidationError) as exc:
            self.auth_view.render_error(str(exc))
            return
        try:
            self.auth_service.sign_out()
        except Exception:
            pass
        clear_auth_state(self.state)
        self.state.pop("history_offset", None)
        self.auth_view.render_success(
            "Password updated. Sign in with your new password."
        )

    def _logout(self) -> None:
        try:
            self.auth_service.sign_out()
        except Exception:
            pass
        finally:
            clear_auth_state(self.state)
            self.state.pop("history_offset", None)
        st.rerun()

    def _run_history(self, session: AuthSession) -> None:
        stored_offset = self.state.get("history_offset", 0)
        if (
            type(stored_offset) is not int
            or stored_offset < 0
            or stored_offset % HistoryRepository.PAGE_SIZE != 0
        ):
            offset = 0
            self.state["history_offset"] = offset
        else:
            offset = stored_offset
        try:
            page = self.history_repository.list_page(
                session.user.id, offset=offset
            )
        except HistoryError as exc:
            self.history_view.render_error(str(exc))
            return
        next_offset = self.history_view.render(
            page.items, offset, page.has_more
        )
        if next_offset is not None:
            self.state["history_offset"] = next_offset
            st.rerun()

    def _run_analysis(self, session: AuthSession) -> None:
        classifier_or_error = self.classifier_loader()
        if isinstance(classifier_or_error, str):
            self.view.render_error(
                "Failed to load model from `./best_model/`. "
                "Run the research notebook first to export a model.\n\n"
                f"Error: {classifier_or_error}"
            )
            return
        classifier = classifier_or_error
        self.view.render_model_info(classifier.meta)

        def handle_image_upload(image):
            with st.spinner("Extracting text with OCR..."):
                try:
                    return extract_text_from_image(image)
                except Exception as exc:
                    self.view.render_error(str(exc))
                    return ""

        text, input_source, is_invalid = self.view.render_input_section(
            on_image_uploaded=handle_image_upload
        )

        def handle_analyze():
            if is_invalid:
                return
            if not text.strip():
                self.view.render_warning(
                    "Please provide a job description to analyze."
                )
                return
            with st.status(
                "Analyzing job description...", expanded=True
            ) as status:
                st.write("⏳ Reading and parsing text...")
                time.sleep(0.5)
                st.write("⏳ Cleaning HTML tags and URLs...")
                cleaned_text = clean_text(text)
                time.sleep(0.5)
                st.write("⏳ Tokenizing input for Transformer model...")
                time.sleep(0.7)
                st.write("⏳ Running sequence classification...")
                label, confidence = classifier.classify_text(cleaned_text)
                time.sleep(1.0)
                st.write("⏳ Extracting heuristics & red flags...")
                red_flags = check_red_flags(text)
                time.sleep(0.5)
                status.update(
                    label="✅ Analysis Complete!",
                    state="complete",
                    expanded=False,
                )

            self.view.render_classification_result(
                label, confidence, red_flags
            )
            try:
                self.history_repository.create(
                    AnalysisHistoryCreate(
                        user_id=session.user.id,
                        input_text=text,
                        input_source=input_source,
                        prediction_label=label,
                        confidence=confidence,
                        red_flags=red_flags,
                    )
                )
            except HistoryError:
                self.view.render_warning(
                    "Analysis history could not be saved."
                )

        self.view.render_result_section(
            is_disabled=is_invalid, on_analyze=handle_analyze
        )
