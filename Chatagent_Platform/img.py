import base64
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Model name MUST be latest
model = genai.GenerativeModel("gemini-pro-vision")

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def ocr_image_with_gemini(image_path: str) -> str:
    try:
        image_bytes = open(image_path, "rb").read()
        response = model.generate_content(
            [
                "Extract text from this image (OCR):",
                {
                    "mime_type": "image/png",  # Adjust if it's jpg/jpeg
                    "data": image_bytes
                }
            ]
        )
        return response.text
    except Exception as e:
        return f"❌ Gemini Vision Error: {e}"
