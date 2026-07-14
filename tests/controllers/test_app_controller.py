import inspect
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

from src.controllers.app_controller import (
    MODEL_LOADING_MESSAGE,
    AppController,
)
from src.models.auth_service import (
    AuthError,
    AuthSession,
    AuthenticatedUser,
)
from src.models.history_repository import HistoryError, HistoryPage
from src.models.session_store import (
    is_model_loading_pending,
    is_recovery_mode,
    load_auth_tokens,
    mark_model_loading_pending,
    pop_auth_notice,
    save_auth_session,
    set_auth_notice,
)
from src.views.auth_view import AuthAction


def session():
    return AuthSession(
        user=AuthenticatedUser(
            id="user-a",
            email="person@example.com",
            full_name="Person A",
        ),
        access_token="access-a",
        refresh_token="refresh-a",
    )


class AppControllerTests(unittest.TestCase):
    def build_controller(self, state=None, query_params=None):
        self.view = Mock()
        self.view.content_container.return_value = nullcontext()
        self.view.render_sidebar.return_value = ("Analyze", False)
        self.view.render_input_section.return_value = ("", "text", False)
        self.view.render_model_loading.return_value = Mock()
        self.view.render_model_load_error.return_value = False
        self.auth_view = Mock()
        self.auth_view.render_auth_page.return_value = None
        self.history_view = Mock()
        self.auth_service = Mock()
        self.history_repository = Mock()
        self.classifier_loader = Mock()
        self.classifier_cache_clearer = Mock()
        self.query_params = (
            query_params if query_params is not None else {}
        )
        return AppController(
            view=self.view,
            auth_view=self.auth_view,
            history_view=self.history_view,
            auth_service=self.auth_service,
            history_repository=self.history_repository,
            classifier_loader=self.classifier_loader,
            classifier_cache_clearer=self.classifier_cache_clearer,
            state=state if state is not None else {},
            query_params=self.query_params,
        )

    def test_anonymous_user_never_loads_classifier(self):
        controller = self.build_controller()

        controller.run()

        self.view.content_container.assert_called_once_with()
        self.auth_view.render_auth_page.assert_called_once_with()
        self.classifier_loader.assert_not_called()

    def test_anonymous_run_renders_pending_notice_before_login(self):
        state = {}
        set_auth_notice(
            state,
            "Password updated. Sign in with your new password.",
        )
        controller = self.build_controller(state)

        controller.run()

        self.auth_view.render_success.assert_called_once_with(
            "Password updated. Sign in with your new password."
        )
        self.auth_view.render_auth_page.assert_called_once_with()
        self.assertIsNone(pop_auth_notice(state))
        self.classifier_loader.assert_not_called()

    def test_auth_notice_is_not_repeated_on_later_anonymous_run(self):
        state = {}
        set_auth_notice(
            state,
            "Password updated. Sign in with your new password.",
        )
        first_controller = self.build_controller(state)
        first_controller.run()

        second_controller = self.build_controller(state)
        second_controller.run()

        self.auth_view.render_success.assert_not_called()
        self.auth_view.render_auth_page.assert_called_once_with()
        self.assertIsNone(pop_auth_notice(state))

    @patch("src.controllers.app_controller.st")
    def test_login_stores_session_marks_loading_and_reruns(self, streamlit):
        state = {}
        controller = self.build_controller(state)
        self.auth_service.sign_in.return_value = session()
        self.auth_view.render_auth_page.return_value = AuthAction(
            "login",
            {
                "email": "person@example.com",
                "password": "password-123",
            },
        )

        controller.run()

        self.assertEqual(load_auth_tokens(state), ("access-a", "refresh-a"))
        self.assertTrue(is_model_loading_pending(state))
        self.assertEqual(state["history_offset"], 0)
        streamlit.rerun.assert_called_once_with()

    def test_authenticated_cold_start_replaces_auth_with_loading_panel(self):
        state = {}
        save_auth_session(state, session())
        mark_model_loading_pending(state, True)
        controller = self.build_controller(state)
        self.auth_service.restore_session.return_value = session()
        classifier = Mock(meta={})
        self.classifier_loader.return_value = classifier
        loading_panel = self.view.render_model_loading.return_value

        controller.run()

        self.auth_view.render_auth_page.assert_not_called()
        self.view.render_model_loading.assert_called_once_with(
            MODEL_LOADING_MESSAGE
        )
        self.classifier_loader.assert_called_once_with()
        loading_panel.empty.assert_called_once_with()
        self.assertFalse(is_model_loading_pending(state))

    def test_classifier_error_shows_retry_without_clearing_authentication(self):
        state = {}
        save_auth_session(state, session())
        mark_model_loading_pending(state, True)
        controller = self.build_controller(state)
        self.auth_service.restore_session.return_value = session()
        self.classifier_loader.return_value = "download failed"

        controller.run()

        self.view.render_model_load_error.assert_called_once_with(
            "download failed"
        )
        self.assertEqual(load_auth_tokens(state), ("access-a", "refresh-a"))
        self.assertFalse(is_model_loading_pending(state))
        self.auth_view.render_auth_page.assert_not_called()
        self.classifier_cache_clearer.assert_not_called()

    @patch("src.controllers.app_controller.st")
    def test_model_retry_clears_only_classifier_cache_and_reruns(
        self, streamlit
    ):
        state = {}
        save_auth_session(state, session())
        controller = self.build_controller(state)
        self.auth_service.restore_session.return_value = session()
        self.classifier_loader.return_value = "download failed"
        self.view.render_model_load_error.return_value = True

        controller.run()

        self.classifier_cache_clearer.assert_called_once_with()
        self.assertTrue(is_model_loading_pending(state))
        self.assertEqual(load_auth_tokens(state), ("access-a", "refresh-a"))
        streamlit.rerun.assert_called_once_with()

    def test_history_page_does_not_load_classifier(self):
        state = {}
        save_auth_session(state, session())
        mark_model_loading_pending(state, True)
        controller = self.build_controller(state)
        self.auth_service.restore_session.return_value = session()
        self.view.render_sidebar.return_value = ("History", False)
        self.history_repository.list_page.return_value = HistoryPage([], False)
        self.history_view.render.return_value = None

        controller.run()

        self.classifier_loader.assert_not_called()
        self.assertFalse(is_model_loading_pending(state))

    @patch("src.controllers.app_controller.st")
    @patch(
        "src.controllers.app_controller.clean_text",
        return_value="cleaned",
    )
    @patch(
        "src.controllers.app_controller.check_red_flags",
        return_value=[],
    )
    def test_history_failure_keeps_rendered_result(
        self,
        _flags,
        _clean,
        streamlit,
    ):
        state = {}
        save_auth_session(state, session())
        controller = self.build_controller(state)
        self.auth_service.restore_session.return_value = session()
        classifier = Mock(meta={})
        classifier.classify_text.return_value = ("Legitimate Job", 0.9)
        self.classifier_loader.return_value = classifier
        streamlit.status.return_value.__enter__.return_value = Mock()
        self.view.render_input_section.return_value = (
            "Job description",
            "text",
            False,
        )
        self.view.render_result_section.side_effect = (
            lambda is_disabled, on_analyze: on_analyze()
        )
        self.history_repository.create.side_effect = HistoryError("save failed")

        controller.run()

        self.view.render_classification_result.assert_called_once()
        self.view.render_warning.assert_called_with(
            "Analysis history could not be saved."
        )

    def test_analysis_flow_contains_no_artificial_sleep(self):
        source = inspect.getsource(AppController._run_analysis)

        self.assertNotIn("time.sleep", source)
        self.assertNotIn("sleep(", source)

    def test_recovery_callback_enters_change_password_mode(self):
        state = {}
        params = {
            "token_hash": "secret-token",
            "type": "recovery",
        }
        controller = self.build_controller(state, params)
        self.auth_service.verify_recovery_token.return_value = session()
        self.auth_service.restore_session.return_value = session()
        self.auth_view.render_recovery_form.return_value = None

        controller.run()

        self.auth_service.verify_recovery_token.assert_called_once_with(
            "secret-token"
        )
        self.assertEqual(params, {})
        self.assertTrue(is_recovery_mode(state))
        self.auth_view.render_recovery_form.assert_called_once_with()
        self.auth_view.render_auth_page.assert_not_called()
        self.classifier_loader.assert_not_called()

    def test_invalid_recovery_callback_is_cleared_and_shows_safe_error(self):
        state = {}
        params = {
            "token_hash": "expired-token",
            "type": "recovery",
        }
        controller = self.build_controller(state, params)
        self.auth_service.verify_recovery_token.side_effect = AuthError(
            "provider detail"
        )

        controller.run()

        self.assertEqual(params, {})
        self.assertFalse(is_recovery_mode(state))
        self.auth_view.render_error.assert_called_once_with(
            "This password recovery link is invalid or has expired."
        )
        self.auth_view.render_auth_page.assert_called_once_with()
        self.classifier_loader.assert_not_called()

    @patch("src.controllers.app_controller.st")
    def test_successful_password_update_stores_notice_and_reruns(
        self,
        streamlit,
    ):
        state = {}
        save_auth_session(state, session())
        state["supabase_recovery_mode"] = True
        controller = self.build_controller(state)
        self.auth_service.restore_session.return_value = session()
        self.auth_view.render_recovery_form.return_value = AuthAction(
            "update_password",
            {
                "password": "new-password",
                "confirmation": "new-password",
            },
        )

        controller.run()

        self.auth_service.update_password.assert_called_once_with(
            "new-password",
            "new-password",
        )
        self.auth_service.sign_out.assert_called_once_with()
        self.assertIsNone(load_auth_tokens(state))
        self.assertFalse(is_recovery_mode(state))
        self.assertEqual(
            pop_auth_notice(state),
            "Password updated. Sign in with your new password.",
        )
        self.auth_view.render_success.assert_not_called()
        streamlit.rerun.assert_called_once_with()
        self.classifier_loader.assert_not_called()

    @patch("src.controllers.app_controller.st")
    def test_failed_password_update_keeps_recovery_mode_without_notice_or_rerun(
        self,
        streamlit,
    ):
        state = {}
        save_auth_session(state, session())
        state["supabase_recovery_mode"] = True
        controller = self.build_controller(state)
        self.auth_service.restore_session.return_value = session()
        self.auth_service.update_password.side_effect = AuthError(
            "Password could not be updated. Try again."
        )
        self.auth_view.render_recovery_form.return_value = AuthAction(
            "update_password",
            {
                "password": "new-password",
                "confirmation": "new-password",
            },
        )

        controller.run()

        self.assertTrue(is_recovery_mode(state))
        self.assertEqual(load_auth_tokens(state), ("access-a", "refresh-a"))
        self.assertIsNone(pop_auth_notice(state))
        self.auth_view.render_error.assert_called_once_with(
            "Password could not be updated. Try again."
        )
        self.auth_service.sign_out.assert_not_called()
        streamlit.rerun.assert_not_called()

    def test_expired_recovery_session_clears_state_without_loading_classifier(
        self,
    ):
        state = {}
        save_auth_session(state, session())
        state["supabase_recovery_mode"] = True
        controller = self.build_controller(state)
        self.auth_service.restore_session.side_effect = AuthError(
            "provider detail"
        )

        controller.run()

        self.assertIsNone(load_auth_tokens(state))
        self.assertFalse(is_recovery_mode(state))
        self.auth_view.render_error.assert_called_once_with(
            "Recovery link or session has expired. Request a new password reset."
        )
        self.auth_view.render_recovery_form.assert_not_called()
        self.classifier_loader.assert_not_called()

    def test_forgot_password_message_remains_generic(self):
        controller = self.build_controller()
        self.auth_service.request_password_reset.side_effect = AuthError(
            "provider detail"
        )
        self.auth_view.render_auth_page.return_value = AuthAction(
            "forgot_password",
            {"email": "person@example.com"},
        )

        controller.run()

        self.auth_view.render_success.assert_called_once_with(
            "If an account exists for that email, a reset message will be sent."
        )
        self.auth_view.render_error.assert_not_called()

    def test_malformed_history_offset_is_reset_to_zero(self):
        state = {"history_offset": "broken"}
        save_auth_session(state, session())
        controller = self.build_controller(state)
        self.auth_service.restore_session.return_value = session()
        self.view.render_sidebar.return_value = ("History", False)
        self.history_repository.list_page.return_value = HistoryPage([], False)
        self.history_view.render.return_value = None

        controller.run()

        self.assertEqual(state["history_offset"], 0)
        self.history_repository.list_page.assert_called_once_with(
            "user-a",
            offset=0,
        )

    @patch("src.controllers.app_controller.st")
    def test_logout_clears_local_tokens_when_remote_signout_fails(
        self,
        streamlit,
    ):
        state = {"history_offset": 20}
        save_auth_session(state, session())
        controller = self.build_controller(state)
        self.auth_service.restore_session.return_value = session()
        self.auth_service.sign_out.side_effect = RuntimeError(
            "provider detail"
        )
        self.view.render_sidebar.return_value = ("Analyze", True)

        controller.run()

        self.assertIsNone(load_auth_tokens(state))
        self.assertNotIn("history_offset", state)
        self.classifier_loader.assert_not_called()
        streamlit.rerun.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
