import google.generativeai as genai
import os

def get_gemini_api_key():
    api_key = "AIzaSyCFrcuthRepY_ihcvUVOgPn6C6j0ZRpq3Q"
    
    if api_key:
        return api_key

def configure_gemini():
    api_key = get_gemini_api_key()
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(
        'gemini-2.5-pro',
    )
    
    return model

gem_model = configure_gemini()