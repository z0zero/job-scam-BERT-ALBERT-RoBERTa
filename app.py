"""
Job Scam Detection - Streamlit Web Application (MVC Architecture)

Entry point for the application. All logic is delegated to the AppController.
"""

from src.controllers.app_controller import AppController

if __name__ == "__main__":
    controller = AppController()
    controller.run()
