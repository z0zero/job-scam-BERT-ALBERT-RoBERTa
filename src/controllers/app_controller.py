import time
import streamlit as st
from src.models.classifier import ScamClassifier
from src.models.preprocessor import clean_text
from src.models.ocr_engine import extract_text_from_image
from src.models.heuristics import check_red_flags
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
                with st.status("Analyzing job description...", expanded=True) as status:
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
                    
                    status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                    
                self.view.render_classification_result(label, confidence, red_flags)

        self.view.render_result_section(on_analyze=handle_analyze)
