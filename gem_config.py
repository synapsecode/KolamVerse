import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load Environment Variables (may include a UTF-8 BOM if file saved with BOM)
load_dotenv()

def get_gemini_api_key():
    api_key = os.environ.get('GEMINI_API_KEY')
    _BOM = "\ufeff" #FOR PowerShell & Windows Use-Cases
    if not api_key:
        # Fallback in case the key name itself has a BOM prefix due to .env BOM parsing
        api_key = os.environ.get(f'{_BOM}GEMINI_API_KEY')
    if api_key:
        return api_key.lstrip(_BOM).strip()
    return None

def configure_gemini():
    api_key = get_gemini_api_key()
    if not api_key:
        # Defer raising; callers can decide to fall back to offline narration
        return None
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    return model
