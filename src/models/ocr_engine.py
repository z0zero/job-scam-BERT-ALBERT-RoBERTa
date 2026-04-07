import os
import pytesseract

# Configure Tesseract OCR path for Windows
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

def extract_text_from_image(image):
    """Extract text from an uploaded image using OCR."""
    try:
        return pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError:
        raise Exception(
            "Tesseract OCR is not installed. "
            "Please install it from https://github.com/UB-Mannheim/tesseract/wiki "
            "and add it to your PATH."
        )
