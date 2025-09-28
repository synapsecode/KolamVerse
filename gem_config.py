import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load Environment Variables (may include a UTF-8 BOM if file saved with BOM)
load_dotenv()

_BOM = "\ufeff"

def get_gemini_api_key():
    """Return the Gemini API key from environment, stripping any UTF-8 BOM.

    Some Windows editors or PowerShell redirections can leave a UTF-8 BOM at the start
    of the .env file. python-dotenv might then register the key name as '\ufeffGEMINI_API_KEY'.
    This helper attempts both, and strips a BOM if it appears in the value.
    """
    api_key = os.environ.get('GEMINI_API_KEY')
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
