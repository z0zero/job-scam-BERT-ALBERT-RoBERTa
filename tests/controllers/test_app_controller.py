import unittest
from unittest.mock import Mock, patch

from src.controllers.app_controller import AppController
from src.models.auth_service import AuthError, AuthSession, AuthenticatedUser
from src.models.history_repository import HistoryError, HistoryPage
from src.models.session_store import (
    is_recovery_mode,
    load_auth_tokens,
    save_auth_session,
)


def session():
    return AuthSession(
        user=AuthenticatedUser(
            id="user-a", email="person@example.com", full_name="Person A"
        ),
        access_token="access-a",
        refresh_token="refresh-a",
    )


class AppControllerTests(unittest.TestCase):
    def build_controller(self, state=None, query_params=None):
        self.view = Mock()
        self.view.render_sidebar.return_value = ("Analyze", False)
        self.view.render_input_section.return_value = ("", "text", False)
        self.auth_view = Mock()
        self.auth_view.render_auth_page.return_value = None
        self.history_view = Mock()
        self.auth_service = Mock()
        self.history_repository = Mock()
        self.classifier_loader = Mock()
        self.query_params = query_params if query_params is not None else {}
        return AppController(
            view=self.view,
            auth_view=self.auth_view,
            history_view=self.history_view,
            auth_service=self.auth_service,
            history_repository=self.history_repository,
            classifier_loader=self.classifier_loader,
            state=state if state is not None else {},
            query_params=self.query_params,
        )

    def test_anonymous_user_never_loads_classifier(self):
        controller = self.build_controller()
        controller.run()
        self.auth_view.render_auth_page.assert_called_once()
        self.classifier_loader.assert_not_called()

    def test_authenticated_user_loads_classifier_on_analyze(self):
        state = {}
        save_auth_session(state, session())
        controller = self.build_controller(state)
        self.auth_service.restore_session.return_value = session()
        self.classifier_loader.return_value = Mock(meta={})

        controller.run()

        self.classifier_loader.assert_called_once()

    @patch("src.controllers.app_controller.st")
    @patch("src.controllers.app_controller.time.sleep")
    @patch("src.controllers.app_controller.clean_text", return_value="cleaned")
    @patch("src.controllers.app_controller.check_red_flags", return_value=[])
    def test_history_failure_keeps_rendered_result(
        self, _flags, _clean, _sleep, streamlit
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
            "Job description", "text", False
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

    def test_callback_is_consumed_and_recovery_never_loads_classifier(self):
        state = {}
        params = {"token_hash": "secret-token", "type": "recovery"}
        controller = self.build_controller(state, params)
        self.auth_service.verify_token.return_value = session()
        self.auth_view.render_recovery_form.return_value = None

        controller.run()

        self.auth_service.verify_token.assert_called_once_with(
            "secret-token", "recovery"
        )
        self.assertEqual(params, {})
        self.assertTrue(is_recovery_mode(state))
        self.classifier_loader.assert_not_called()

    def test_recovery_rerun_restores_session_before_updating_password(self):
        state = {}
        callback_params = {"token_hash": "secret-token", "type": "recovery"}
        callback_controller = self.build_controller(state, callback_params)
        self.auth_service.verify_token.return_value = session()
        self.auth_service.restore_session.return_value = session()
        self.auth_view.render_recovery_form.return_value = None
        callback_controller.run()

        recovery_service = Mock()
        recovery_service.restore_session.return_value = session()
        recovery_view = Mock()
        recovery_view.render_recovery_form.return_value = Mock(
            payload={"password": "new-password", "confirmation": "new-password"}
        )
        classifier_loader = Mock()
        rerun_controller = AppController(
            view=self.view,
            auth_view=recovery_view,
            history_view=self.history_view,
            auth_service=recovery_service,
            history_repository=self.history_repository,
            classifier_loader=classifier_loader,
            state=state,
            query_params={},
        )

        rerun_controller.run()

        self.assertEqual(
            recovery_service.method_calls[:2],
            [
                unittest.mock.call.restore_session("access-a", "refresh-a"),
                unittest.mock.call.update_password(
                    "new-password", "new-password"
                ),
            ],
        )
        classifier_loader.assert_not_called()

    def test_expired_recovery_session_clears_state_without_loading_classifier(self):
        state = {}
        save_auth_session(state, session())
        state["supabase_recovery_mode"] = True
        controller = self.build_controller(state)
        self.auth_service.restore_session.side_effect = AuthError("provider detail")

        controller.run()

        self.assertIsNone(load_auth_tokens(state))
        self.assertFalse(is_recovery_mode(state))
        self.auth_view.render_error.assert_called_once_with(
            "Recovery link or session has expired. Request a new password reset."
        )
        self.auth_view.render_recovery_form.assert_not_called()
        self.classifier_loader.assert_not_called()

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
            "user-a", offset=0
        )

    @patch("src.controllers.app_controller.st")
    def test_logout_clears_local_tokens_when_remote_signout_fails(self, streamlit):
        state = {"history_offset": 20}
        save_auth_session(state, session())
        controller = self.build_controller(state)
        self.auth_service.restore_session.return_value = session()
        self.auth_service.sign_out.side_effect = RuntimeError("provider detail")
        self.view.render_sidebar.return_value = ("Analyze", True)

        controller.run()

        self.assertIsNone(load_auth_tokens(state))
        self.assertNotIn("history_offset", state)
        self.classifier_loader.assert_not_called()
        streamlit.rerun.assert_called_once()


if __name__ == "__main__":
    unittest.main()
