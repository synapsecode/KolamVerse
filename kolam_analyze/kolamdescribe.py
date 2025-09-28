import hashlib
import os
import shutil
import sys
from typing import Optional
from .kolam_semantics import semantic_json_string
from .narration_cache import AINarrationCache
from .offline_narrator import offline_narrate
from .utils import compress_semantics, render_path_png
from .kolam_analyzer import analyze_kolam_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gem_config import configure_gemini

def describe_kolam_characteristics(fileobject, filepath):
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(fileobject, buffer)
        # Analyze the kolam image
        analysis_result = analyze_kolam_image(filepath)
        return (analysis_result, None)
    except Exception as e:
        return (None, str(e))
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass

def convert_eulerian_path_to_semantics(path, compress=True):
    semantics = semantic_json_string(path)
    semantics_ai = compress_semantics(semantics) if compress else semantics
    sem_hash = hashlib.sha256(semantics_ai.encode("utf-8")).hexdigest()
    return (semantics, semantics_ai, sem_hash)

async def describe_kolam_using_ai(
    eulerian_path,
    include_image: bool = True,
    compress: bool = True,
    mode: str = "auto"  # auto | offline | ai
):
    try:
        semantics, semantics_ai, sem_hash = convert_eulerian_path_to_semantics(eulerian_path, compress)

        # ---- Step 4: Cached narration? ----
        if await AINarrationCache.contains(sem_hash)  and mode != "offline":
            return success_response(
                narration=await AINarrationCache.get(sem_hash),
                semantics=semantics,
                cached=True,
                source="ai-cache",
                ai_used=True
            )

        # ---- Step 5: Offline only ----
        if mode == "offline":
            return success_response(
                narration=offline_narrate(semantics_ai),
                semantics=semantics,
                source="offline",
                ai_used=False
            )

        # ---- Step 6: AI narration attempt ----
        img_b64 = render_path_png(eulerian_path) if include_image else None
        try:
            narration = generate_kolam_narration(semantics_ai, img_b64)
            await AINarrationCache.add(key=sem_hash, value=narration)
            return success_response(
                narration=narration,
                semantics=semantics,
                cached=False,
                source="ai",
                ai_used=True
            )
        except Exception as e:
            return handle_ai_error(e, semantics_ai, semantics)

    except Exception as e:
        return (None, str(e))


# ---------------- Helpers ---------------- #

def success_response(narration, semantics, **extra):
    return ({
        "success": True,
        "narration": narration,
        "semantics": semantics,
        **extra
    }, None)


def offline_fallback(semantics_ai, semantics, note="offline fallback"):
    narration = offline_narrate(semantics_ai),
    return success_response(
        narration=narration,
        semantics=semantics,
        source="offline-fallback",
        ai_used=False,
        note=note
    )

def handle_ai_error(exc, semantics_ai, semantics):
    msg = str(exc)
    quota = ("quota" in msg.lower()) or ("429" in msg)
    if quota:
        return offline_fallback(
            semantics_ai, semantics,
            "AI quota exceeded or rate limited; offline narration provided",
        )
    return offline_fallback(
        semantics_ai, semantics,
        f"AI error: {msg[:160]} (offline narration substituted)"
    )


def generate_kolam_narration(structured_json: str, final_image_b64: Optional[str] = None) -> str:
    """Call Gemini model to produce a narrated description of drawing order."""
    client = configure_gemini()

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

    response = client.generate_content(parts)
    return (getattr(response, "text", None) or "Kolam drawing description unavailable.").strip()

