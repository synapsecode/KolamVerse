import os
from typing import Optional

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore


DEFAULT_PREFERRED_MODELS = [
    # Newer naming conventions first
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro",
]


class GeminiNotConfigured(Exception):
    pass


def _configure():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise GeminiNotConfigured("GEMINI_API_KEY not set in environment")
    if genai is None:
        raise GeminiNotConfigured("google-generativeai package not installed")
    genai.configure(api_key=api_key)


_cached_model_name: str | None = None


def _select_model() -> str:
    global _cached_model_name
    if _cached_model_name:
        return _cached_model_name

    override = os.getenv("GEMINI_MODEL")
    model_candidates = ([override] if override else []) + DEFAULT_PREFERRED_MODELS

    try:
        models = list(genai.list_models())  # type: ignore
    except Exception as e:
        # Fallback to first candidate if listing fails
        _cached_model_name = model_candidates[0]
        return _cached_model_name

    # Filter models that support generateContent
    usable = {}
    for m in models:
        try:
            name = getattr(m, "name", "")
            gen_methods = getattr(m, "supported_generation_methods", [])
            if name and any("generateContent" in gm or gm == "generateContent" for gm in gen_methods):
                usable[name] = True
        except Exception:
            continue

    for cand in model_candidates:
        # API returns names like models/gemini-1.5-flash-latest
        direct = cand if cand.startswith("models/") else f"models/{cand}"
        if direct in usable:
            _cached_model_name = cand  # store canonical short form
            return cand

    # Last resort: pick any usable model
    if usable:
        picked = list(usable.keys())[0].replace("models/", "")
        _cached_model_name = picked
        return picked

    raise GeminiNotConfigured("No suitable Gemini model supporting generateContent found.")


def generate_kolam_narration(structured_json: str, final_image_b64: Optional[str] = None) -> str:
    """Call Gemini model to produce a narrated description of drawing order."""
    _configure()
    model_name = _select_model()
    model = genai.GenerativeModel(model_name)

    system_preamble = (
        "You are an expert in traditional South Indian kolam (rangoli) pattern narration. "
        "Given a structured JSON describing ordered drawing steps, produce a concise, tutorial style narrative. "
        "Rules: 1) Start with an overview. 2) Group steps into phases. 3) Mention symmetry & loops. "
        "4) Avoid inventing colors/materials. 5) Keep under 230 words."
    )

    parts = [
        {"text": system_preamble},
        {"text": "\nStructured JSON describing steps:\n" + structured_json}
    ]
    if final_image_b64:
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": final_image_b64
            }
        })

    response = model.generate_content(parts)
    return (getattr(response, "text", None) or "Kolam drawing description unavailable.").strip()


__all__ = ["generate_kolam_narration", "GeminiNotConfigured"]
