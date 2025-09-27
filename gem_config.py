import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

def get_gemini_api_key():
    api_key = os.environ.get('GEMINI_API_KEY')
    
    if api_key:
        return api_key

def configure_gemini():
    api_key = get_gemini_api_key()
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(
        'gemini-2.5-pro',
    )
    
    return model