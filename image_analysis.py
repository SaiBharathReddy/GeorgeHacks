import streamlit as st
from PIL import Image
from openai import OpenAI
from PIL import Image
import requests
import base64
from io import BytesIO
from dotenv import load_dotenv
from googletrans import LANGUAGES
from deep_translator import GoogleTranslator
import pytesseract

load_dotenv()


def vision_llm_initialiation(provider: str):
    @st.cache_resource
    def _get_client(provider_name):
        api_key = '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'
        model_name = "google/gemini-2.0-pro-exp-02-05:free"
        if provider == "gemini":
            from openai import OpenAI
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            model_name = "google/gemini-2.0-pro-exp-02-05:free"
            return client, model_name
        else:
            raise ValueError("Unsupported provider")

    return _get_client(provider)


def extract_text_from_image(image: Image) -> str:
    """Extracts text from image using Tesseract OCR"""
    return pytesseract.image_to_string(image)

def generate_response(image: Image, client, model_name: str) -> str:

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    system_prompt = """
    Analyze the uploaded medical document as a medical expert in the industry and extract dont give long Outputs The total words should be less than 500:

1. **Patient Summary**  
   - Name/Age/Gender
   - Primary diagnosis
   - Key symptoms
   - Date of diagnosis/incident

2. **Incident Details** (if trauma/accident-related)
   - Type of injury
   - Affected body areas
   - Severity assessment

3. **Treatment Plan**  
   [✓] List prescribed medications with dosages  
   [✓] Identify prescribed therapies/procedures  
   [✓] Note treatment duration

4. **Precautions**  
   - Activity restrictions
   - Dietary modifications
   - Warning signs requiring urgent care

Present findings in clear bullet points using non-technical language. Highlight critical information in **bold**. Omit complex medical jargon unless essential.    """

    try:
        extracted_text = extract_text_from_image(image)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": extracted_text}
]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )

        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None




def generate_response1(image: Image, client, model_name: str, target_language='en') -> str:
    # buffered = BytesIO()
    # image.save(buffered, format="JPEG")
    # img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    system_prompt = """
        You are a highly knowledgeable medical AI assistant. Your task is to analyze an uploaded image that contains a summary of a patient's medical condition. From this image, extract crucial health information including symptoms, diagnosis, and prescribed treatments. 

    1. **Summarize Key Findings**: Provide a concise summary of the patient's medical condition as described in the image. Focus on major symptoms, diagnosis, and any critical information that a patient must understand about their health status.

    2. **Evaluate and Explain Treatments**: Review the listed medications and treatments in the image. Verify their appropriateness for the diagnosed conditions based on current medical guidelines. Explain why each medication or treatment is recommended, and discuss any potential side effects or important considerations for their use.

    3. **Suggest Remedies and Precautions**: Based on the diagnosis and treatments identified, suggest any additional home remedies or precautions the patient should take. This could include lifestyle adjustments, dietary changes, or activities to avoid.

    4. **Check for Errors**: Critically assess if there are any inconsistencies or errors in the medical information presented in the image. This could include mismatched treatments for the diagnosed conditions or outdated therapies.

    Ensure that your responses are clear, medically accurate, and easy for a layperson to understand. Use simple language to explain complex medical terms and concepts. Your goal is to ensure that the patient fully understands their medical situation, how to manage it, and any necessary actions they should take regarding their treatment plan.
        """

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image_url",
                                          "image_url": img_str}]}
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        original_text = response.choices[0].message.content
        if target_language != 'en':  # Assuming English is the default
            return GoogleTranslator(source='auto', target='hi').translate(text=original_text)
        return original_text
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

def image_analysis_section():
    st.write("## Image Upload Section")
    st.write("Upload an image and submit it for future analysis.")

    # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Initialize client and model once
        client, model_name = vision_llm_initialiation("gemini")

        # Language selection
        langs_dict = GoogleTranslator().get_supported_languages(as_dict=True)
        sorted_langs = sorted(langs_dict.items(), key=lambda item: item[1])
        target_language = st.selectbox(
            "Select the target language for translation:",
            options=sorted_langs,
            format_func=lambda x: f"{x[1]} ({x[0]})"  # Display format "English (en)"
        )

        if st.button("Submit Image"):
            st.success("Image submitted successfully!")
            response = generate_response(image, client, model_name)
            if response:
                translated_response = GoogleTranslator(source='auto', target=target_language[1]).translate(text=response)  # Assuming translate_text is implemented
                st.write("### AI Generated Medical Advice:")
                st.write(translated_response)
            else:
                st.error("Failed to generate response.")