import streamlit as st
from src.models.classifier import ScamClassifier
from src.models.preprocessor import clean_text
from src.models.ocr_engine import extract_text_from_image
from src.views.main_view import MainView

@st.cache_resource
def get_classifier():
    classifier = ScamClassifier()
    try:
        classifier.load_model()
        return classifier
    except Exception as e:
        return str(e)

class AppController:
    def __init__(self):
        self.view = MainView()
        # Initialize page config as soon as controller is created because st.set_page_config must be the first Streamlit command.
        self.view.setup_page()

    def run(self):
        self.view.render_header()
        
        classifier_or_err = get_classifier()
        if isinstance(classifier_or_err, str):
             self.view.render_error(
                f"Failed to load model from `./best_model/`. "
                f"Run the research notebook first to export a model.\n\nError: {classifier_or_err}"
             )
             return
             
        classifier = classifier_or_err
        self.view.render_model_info(classifier.meta)

        def handle_image_upload(image):
            with st.spinner("Extracting text with OCR..."):
                 try:
                     return extract_text_from_image(image)
                 except Exception as e:
                     self.view.render_error(str(e))
                     return ""

        text = self.view.render_input_section(on_image_uploaded=handle_image_upload)

        def handle_analyze():
            if not text.strip():
                self.view.render_warning("Please provide a job description to analyze.")
            else:
                with st.spinner("Analyzing..."):
                    cleaned_text = clean_text(text)
                    label, confidence = classifier.classify_text(cleaned_text)
                self.view.render_classification_result(label, confidence)

        self.view.render_result_section(on_analyze=handle_analyze)
