import streamlit as st
from PIL import Image

class MainView:
    @staticmethod
    def setup_page():
        st.set_page_config(
            page_title="Job Scam Detector",
            page_icon="shield",
            layout="centered",
        )

    @staticmethod
    def render_header():
        st.title("Job Scam Detector")
        st.markdown(
            "Analyze job postings to detect potential scams using NLP. "
            "Paste a job description or upload a screenshot."
        )

    @staticmethod
    def render_model_info(meta):
        if meta:
            st.info(
                f"**Model:** {meta['model_name']} ({meta['hf_model_id']}) | "
                f"F1: {meta['metrics']['f1']:.4f} | "
                f"Accuracy: {meta['metrics']['accuracy']:.4f}"
            )
        else:
            st.error(
                "Failed to load model metadata. "
                "Ensure the model is loaded correctly."
            )

    @staticmethod
    def render_error(message):
        st.error(message)

    @staticmethod
    def render_warning(message):
        st.warning(message)

    @staticmethod
    def render_input_section(on_image_uploaded=None):
        st.subheader("Input")
        input_mode = st.radio(
            "Choose input method:",
            ["Paste Text", "Upload Image"],
            horizontal=True,
        )

        text = ""
        if input_mode == "Paste Text":
            text = st.text_area(
                "Paste the job description below:",
                height=250,
                placeholder="Enter or paste the full job posting text here...",
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a screenshot of the job posting:",
                type=["png", "jpg", "jpeg"],
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)

                if on_image_uploaded:
                    extracted = on_image_uploaded(image)
                    text = st.text_area(
                        "Extracted text (edit if needed):",
                        value=extracted,
                        height=250,
                    )
        return text

    @staticmethod
    def render_result_section(on_analyze=None):
        st.subheader("Result")
        
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
        if analyze_clicked and on_analyze:
            on_analyze()
            
    @staticmethod
    def render_classification_result(label, confidence, red_flags=None):
        col1, col2 = st.columns(2)
        
        with col1:
            if label == "Legitimate Job":
                st.success(f"**{label}**")
            else:
                st.error(f"**{label}**")
            st.metric(label="Confidence", value=f"{confidence * 100:.1f}%")
            
        with col2:
            num_flags = len(red_flags) if red_flags else 0
            if num_flags > 0:
                st.warning(f"**{num_flags} Red Flag(s) Detected**")
            else:
                st.info("**0 Red Flags**")
                
        if label != "Legitimate Job":
            with st.expander("🔍 Analysis Indicators (Red Flags)", expanded=True):
                if red_flags and len(red_flags) > 0:
                    for flag in red_flags:
                        st.write(f"- ⚠️ {flag}")
                else:
                    st.write("🤖 Model detected suspicious patterns from the training data, although no explicit heuristic red flags were matched.")
