import unittest
from unittest.mock import patch

from src.models.supabase_client import (
    SupabaseConfigError,
    SupabaseSettings,
    create_session_client,
    load_supabase_settings,
)


class SupabaseClientTests(unittest.TestCase):
    def test_load_settings_strips_values(self):
        settings = load_supabase_settings(
            {
                "SUPABASE_URL": " https://project.supabase.co ",
                "SUPABASE_PUBLISHABLE_KEY": " sb_publishable_test ",
                "APP_URL": " http://localhost:8501 ",
            }
        )

        self.assertEqual(
            settings,
            SupabaseSettings(
                url="https://project.supabase.co",
                publishable_key="sb_publishable_test",
                app_url="http://localhost:8501",
            ),
        )

    def test_load_settings_names_missing_secret_without_printing_values(self):
        with self.assertRaisesRegex(
            SupabaseConfigError, "SUPABASE_PUBLISHABLE_KEY"
        ):
            load_supabase_settings(
                {
                    "SUPABASE_URL": "https://project.supabase.co",
                    "APP_URL": "http://localhost:8501",
                }
            )

    @patch("src.models.supabase_client.create_client")
    def test_create_session_client_uses_publishable_key(self, create_client):
        settings = SupabaseSettings(
            url="https://project.supabase.co",
            publishable_key="sb_publishable_test",
            app_url="http://localhost:8501",
        )

        result = create_session_client(settings)

        self.assertIs(result, create_client.return_value)
        create_client.assert_called_once_with(
            "https://project.supabase.co", "sb_publishable_test"
        )


if __name__ == "__main__":
    unittest.main()
