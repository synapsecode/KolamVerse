from typing import Callable
import cv2
import threading

def load_ai_prompt_template():
    """Load the AI prompt template from external file"""
    try:
        with open("prompt_seed_instructions.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback prompt if file is missing
        return """You are an expert designer of Kolam art using a specific L-system. Convert the user's description into a simple L-system axiom using only F, A, B, C commands. Only output the final axiom string.

User Description: "{user_prompt}"
Axiom:"""

class JPEGEncoder:
    _backend = None
    _local = threading.local()

    @classmethod
    def _get_turbo(cls):
        if not hasattr(cls._local, "turbo"):
            from turbojpeg import TurboJPEG
            cls._local.turbo = TurboJPEG()
        return cls._local.turbo

    @classmethod
    def encode(cls, img, quality: int = 30) -> bytes:
        if cls._backend is None:
            try:
                cls._get_turbo()  # Try initializing TurboJPEG
                cls._backend = "turbo"
                print("[JPEGEncoder] Using TurboJPEG")
            except Exception:
                cls._backend = "cv2"
                print("[JPEGEncoder] Falling back to OpenCV")

        if cls._backend == "turbo":
            return cls._get_turbo().encode(img, quality=quality)

        # Fallback to OpenCV
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok, buf = cv2.imencode(".jpg", img, encode_param)
        if not ok:
            raise RuntimeError("OpenCV JPEG encoding failed")
        return buf.tobytes()