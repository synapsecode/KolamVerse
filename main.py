import base64
import os
import shutil
import hashlib
from typing import Optional, Dict
import numpy as np  # Needed for linspace in _compress_semantics
from fastapi import FastAPI, File, Response, UploadFile, Query, Body
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from gem_config import configure_gemini
from kolam2csv import image_to_kolam_csv
from kolam_frame_manger import KolamFrameManager
from kolamanimator import animate_eulerian_stream, compute_eulerian_path, load_all_points, normalize_strokes
from kolamdraw_web import draw_kolam_web_bytes
from utils import load_ai_prompt_template
from kolam_analyzer import analyze_kolam_image
from kolam_semantics import semantic_json_string
from gemini_client import generate_kolam_narration, GeminiNotConfigured
from offline_narrator import offline_narrate
from lsystem import generate_lsystem_state  # for simulate_kolam_path used in seed narration
import json

app = FastAPI()
kolam_frame_manager = KolamFrameManager()

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# --- simple .env loader (no external dependency) ---
def _load_env_file(path: str = ".env"):
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k not in os.environ:  # don't override existing
                    os.environ[k.strip()] = v.strip()
    except Exception:
        pass

_load_env_file()

# In-memory narration cache {hash(semantics_json): narration_text}
_AI_NARRATION_CACHE: Dict[str, str] = {}


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_path)

#  ------------------- KolamTrace ---------------------

@app.get("/kolamtrace", response_class=HTMLResponse)
def app_kolamtrace(): 
    index_path = os.path.join(STATIC_DIR, "kolamtrace.html")
    return FileResponse(index_path, headers={"Cache-Control":"no-store"})

@app.post("/upload_kolam")
async def upload_kolam(file: UploadFile = File(...)):
    # Check content type
    if not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Only image files are allowed"}, status_code=400)

    await kolam_frame_manager.clear()

    # Save uploaded file
    file_path = os.path.join(STATIC_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Convert Image to CSV
    csv_filename = f"{os.path.splitext(file.filename)[0]}.csv"
    csv_path = os.path.join(STATIC_DIR, csv_filename)
    image_to_kolam_csv(file_path, csv_path)

    # Delete the Image
    os.remove(file_path)

    return {"csv_file": csv_filename}


@app.get("/animate_kolam")
async def animate_kolam(csv_file: str = Query(..., description="CSV filename generated from /upload_kolam")):
    csv_path = os.path.join(STATIC_DIR, csv_file)

    if not os.path.exists(csv_path):
        return JSONResponse({"error": "CSV file not found"}, status_code=404)
    
    kolam_frame_manager.clear()

    strokes = load_all_points(csv_path)
    strokes = normalize_strokes(strokes)
    path = compute_eulerian_path(strokes, tol=1e-1)

    # Delete the CSV
    os.remove(csv_path)

    return StreamingResponse(
        animate_eulerian_stream(path, kolam_frame_manager, step_delay=0.00005),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/kolam_snapshots")
async def kolam_snapshots():
    """Return captured animation snapshots.

    While frames are still being generated return 202 to avoid log spam with 404s.
    """
    snapshots = await kolam_frame_manager.get_frames()
    if not snapshots:
        return JSONResponse({"status": "pending", "frames": []}, status_code=202, headers={"Cache-Control":"no-store"})
    frames_b64 = [base64.b64encode(f).decode("utf-8") for f in snapshots]
    return JSONResponse({"status":"ready","frames": frames_b64}, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})

# ---------------- KolamDraw -------------------

@app.get("/kolamdraw", response_class=HTMLResponse)
def app_kolamdraw():
    index_path = os.path.join(STATIC_DIR, "kolamdraw.html")
    return FileResponse(index_path, headers={"Cache-Control":"no-store"})

@app.get("/drawkolam")
def drawkolam(seed: str = "FBFBFBFB", depth: int = 1):
    """Render a kolam from a seed using the web drawing implementation."""
    img_bytes = draw_kolam_web_bytes(seed=seed, depth=depth)
    return Response(content=img_bytes, media_type="image/png")

@app.post("/generate_seed_from_prompt")
async def generate_seed_from_prompt(payload: dict = Body(...)):
    """Generate a seed string via Gemini from a natural language prompt."""
    user_prompt = payload.get("prompt")
    if not user_prompt:
        return JSONResponse({"error": "Prompt cannot be empty"}, status_code=400)
    prompt_template = load_ai_prompt_template()
    instructional_prompt = prompt_template.format(user_prompt=user_prompt)
    try:
        client = configure_gemini()
        if client is None:
            return JSONResponse({"error": "Gemini not configured"}, status_code=503)
        response = client.generate_content(instructional_prompt)
        generated_seed = (response.text or "").strip()
        if not generated_seed:
            return JSONResponse({"error": "Empty response from model"}, status_code=502)
        if all(c in "FABLRC" for c in generated_seed):
            return JSONResponse({"seed": generated_seed})
        return JSONResponse({"error": "Failed to generate a valid seed.", "details": generated_seed}, status_code=422)
    except Exception as e:
        return JSONResponse({"error": "AI generation failed", "details": str(e)}, status_code=500)
    
# ---------------- Kolam Analysis & Description -------------------

@app.post("/describe_kolam")
async def describe_kolam(file: UploadFile = File(...)):
    """Analyze kolam image and return natural language description."""
    # Check content type
    if not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Only image files are allowed", "success": False}, status_code=400)

    # Save uploaded file temporarily
    temp_filename = f"temp_{file.filename}"
    file_path = os.path.join(STATIC_DIR, temp_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Analyze the kolam image
        analysis_result = analyze_kolam_image(file_path)
        
        return JSONResponse({
            "description": analysis_result["description"],
            "features": analysis_result["features"],
            "success": True
        })
    
    except Exception as e:
        return JSONResponse({
            "error": f"Analysis failed: {str(e)}",
            "success": False
        }, status_code=500)
    
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

@app.get("/describe_seed")
def describe_seed(seed: str = "FBFBFBFB"):
    """Generate natural language description for an L-system seed (technical summary)."""
    try:
        description = generate_seed_description(seed)
        return JSONResponse({
            "success": True,
            "seed": seed,
            "description": description
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"Description generation failed: {e}"
        }, status_code=500)

def generate_seed_description(seed):
    """Generate natural language description for L-system seed."""
    from collections import Counter
    
    # Count character occurrences
    char_counts = Counter(seed)
    
    description = f"This L-system seed '{seed}' will generate a kolam pattern with the following characteristics: "
    
    # Analyze seed components
    if char_counts.get('F', 0) > 0:
        description += f"The pattern includes {char_counts['F']} forward line segment{'s' if char_counts['F'] > 1 else ''}. "
    
    if char_counts.get('A', 0) > 0:
        description += f"It contains {char_counts['A']} arc element{'s' if char_counts['A'] > 1 else ''} that create smooth curved sections. "
    
    if char_counts.get('B', 0) > 0:
        description += f"The design incorporates {char_counts['B']} complex arc motif{'s' if char_counts['B'] > 1 else ''} with intricate curved patterns. "
    
    if char_counts.get('L', 0) > 0 or char_counts.get('R', 0) > 0:
        turns = char_counts.get('L', 0) + char_counts.get('R', 0)
        description += f"The pattern includes {turns} directional change{'s' if turns > 1 else ''} that create angular variations. "
    
    # Analyze pattern characteristics
    seed_length = len(seed)
    if seed_length <= 4:
        description += "This is a simple pattern with minimal complexity. "
    elif seed_length <= 8:
        description += "This represents a moderately complex kolam design. "
    else:
        description += "This creates an intricate kolam with high complexity. "

    unique_chars = len(set(seed))
    if unique_chars < seed_length / 2:
        description += "The seed shows repetitive elements that will create rhythmic patterns in the final design. "

    if seed == seed[::-1]:
        description += "The seed is palindromic, which will result in symmetrical kolam patterns. "

    description += "When expanded through L-system iterations, this seed will generate a traditional kolam following the fundamental rules of continuous line drawing around dots."
    return description

@app.get("/narrate_seed")
def narrate_seed(seed: str = "FBFBFBFB", depth: int = 1):
    """Return an offline (non-AI) narrative of the kolam path implied by the seed.

    This avoids the technical L-system description and instead produces a stylistic
    flow narration similar to offline fallback for traced images.
    """
    try:
        # Simulate the path (points only). We'll convert to a pseudo-eulerian 'path list' like [(p_i, p_{i+1})...]
        pts, types = simulate_kolam_path(seed=seed, depth=depth)
        if len(pts) < 2:
            return JSONResponse({"success": False, "error": "Seed produced insufficient points."}, status_code=400)
        path_edges = list(zip(pts[:-1], pts[1:]))
        sem_json = semantic_json_string(path_edges)
        narration = offline_narrate(sem_json)
        return JSONResponse({
            "success": True,
            "seed": seed,
            "depth": depth,
            "narration": narration,
            "source": "offline-seed"
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": f"Seed narration failed: {e}"}, status_code=500)

# ---------------- Seed Path Simulation Helper -------------------
def simulate_kolam_path(seed: str = "FBFBFBFB", depth: int = 1, step: int = 20, angle: int = 0):
    """Lightweight path simulation for seeds (used only for offline seed narration)."""
    from math import cos, sin, radians, sqrt
    from lsystem import generate_lsystem_state
    state = generate_lsystem_state(seed, depth)
    x, y = 0.0, 0.0
    heading = float(angle)
    path_points = [(x, y)]

    def add_arc(x, y, heading, radius, extent_deg, steps=20):
        points = []
        step_angle = radians(extent_deg / steps)
        cur_x, cur_y = x, y
        for i in range(steps):
            theta1 = radians(heading) + i * step_angle
            theta2 = radians(heading) + (i + 1) * step_angle
            x_next = cur_x + radius * (sin(theta2) - sin(theta1))
            y_next = cur_y - radius * (cos(theta2) - cos(theta1))
            points.append((x_next, y_next))
            cur_x, cur_y = x_next, y_next
        heading += extent_deg
        return points, heading, (cur_x, cur_y)

    for ch in state:
        if ch == 'F':
            x2 = x + step * cos(radians(heading))
            y2 = y + step * sin(radians(heading))
            path_points.append((x2, y2))
            x, y = x2, y2
        elif ch == 'A':
            arc_points, heading, (x, y) = add_arc(x, y, heading, radius=step * 1.2, extent_deg=90)
            path_points.extend(arc_points)
        elif ch == 'B':
            I = step / sqrt(2)
            x2 = x + I * cos(radians(heading))
            y2 = y + I * sin(radians(heading))
            path_points.append((x2, y2))
            x, y = x2, y2
            arc_points, heading, (x, y) = add_arc(x, y, heading, radius=I * 1.1, extent_deg=270)
            path_points.extend(arc_points)
            x2 = x + I * cos(radians(heading))
            y2 = y + I * sin(radians(heading))
            path_points.append((x2, y2))
            x, y = x2, y2
        elif ch == 'L':
            heading += 45
        elif ch == 'R':
            heading -= 45
        else:
            pass
    # Convert to (point, point) edge list style elsewhere; return raw points here
    return path_points, []


# ---------------- AI Kolam Narration (Gemini) -------------------

def _render_path_png(path, size: int = 800):
    """Render full path to a PNG (single frame) and return base64 string."""
    import cv2
    import numpy as np
    canvas = np.full((size, size, 3), (46, 95, 59), dtype=np.uint8)
    xs = [p[0] for e in path for p in e]
    ys = [p[1] for e in path for p in e]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale = min(size / (max_x - min_x + 1), size / (max_y - min_y + 1)) * 0.9

    def to_px(pt):
        x = int((pt[0] - min_x) * scale + size * 0.05)
        y = int((pt[1] - min_y) * scale + size * 0.05)
        return (x, size - y)

    for (u, v) in path:
        cv2.line(canvas, to_px(u), to_px(v), (255, 255, 255), 3)

    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _compress_semantics(raw_json: str, max_steps: int = 60) -> str:
    """Reduce semantics JSON size by keeping metadata and sampling steps.

    We parse the JSON manually (defensive) and down-sample steps if needed.
    """
    import json
    try:
        data = json.loads(raw_json)
    except Exception:
        return raw_json  # fallback

    steps = data.get("steps", [])
    total = len(steps)
    if total > max_steps:
        # sample evenly
        indices = set(int(i) for i in np.linspace(0, total - 1, max_steps))  # type: ignore
        sampled = [s for idx, s in enumerate(steps) if idx in indices]
        data["steps"] = sampled
        data["metadata"]["sampled_from"] = total
        data["metadata"]["kept_steps"] = len(sampled)
    return json.dumps(data, separators=(",", ":"))


@app.post("/describe_kolam_ai")
async def describe_kolam_ai(
    file: UploadFile = File(...),
    include_image: bool = True,
    compress: bool = True,
    mode: str = "auto"  # auto|offline|ai
):
    """Generate an AI narrated drawing description using Gemini.

    Falls back with explanatory error if API key / dependency missing.
    """
    if not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Only image files are allowed", "success": False}, status_code=400)

    temp_path = os.path.join(STATIC_DIR, f"ai_{file.filename}")
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Reuse existing pipeline to produce CSV + path
        csv_path = temp_path + ".csv"
        image_to_kolam_csv(temp_path, csv_path)
        strokes = load_all_points(csv_path)
        strokes = normalize_strokes(strokes)
        path = compute_eulerian_path(strokes, tol=1e-1)

        semantics_json = semantic_json_string(path)
        if compress:
            semantics_json_for_ai = _compress_semantics(semantics_json)
        else:
            semantics_json_for_ai = semantics_json

        # Cache key
        sem_hash = hashlib.sha256(semantics_json_for_ai.encode("utf-8")).hexdigest()

        # If already narrated, return cache directly (no API hit)
        if sem_hash in _AI_NARRATION_CACHE and mode != "offline":
            return JSONResponse({
                "success": True,
                "narration": _AI_NARRATION_CACHE[sem_hash],
                "semantics": semantics_json,
                "cached": True,
                "source": "ai-cache",
                "ai_used": True
            })

        # Offline forced or AI key absent -> offline narration directly
        if mode == "offline":
            offline_text = offline_narrate(semantics_json_for_ai)
            return JSONResponse({
                "success": True,
                "narration": offline_text,
                "semantics": semantics_json,
                "source": "offline",
                "ai_used": False
            })

        img_b64: Optional[str] = _render_path_png(path) if include_image else None

        try:
            narration = generate_kolam_narration(semantics_json_for_ai, img_b64)
            _AI_NARRATION_CACHE[sem_hash] = narration
            return JSONResponse({
                "success": True,
                "narration": narration,
                "semantics": semantics_json,
                "cached": False,
                "source": "ai",
                "ai_used": True
            })
        except GeminiNotConfigured as cfg_err:
            # Treat lack of configuration as a successful offline fallback
            offline_text = offline_narrate(semantics_json_for_ai)
            return JSONResponse({
                "success": True,
                "narration": offline_text,
                "semantics": semantics_json,
                "source": "offline-fallback",
                "ai_used": False,
                "note": f"Gemini not configured: {cfg_err}"
            })
        except Exception as llm_err:
            msg = str(llm_err)
            quota = ("quota" in msg.lower()) or ("429" in msg)
            offline_text = offline_narrate(semantics_json_for_ai)
            if quota:
                return JSONResponse({
                    "success": True,
                    "narration": offline_text,
                    "semantics": semantics_json,
                    "source": "offline-fallback",
                    "ai_used": False,
                    "note": "AI quota exceeded or rate limited; offline narration provided",
                    "retry_hint_seconds": 30
                })
            return JSONResponse({
                "success": True,
                "narration": offline_text,
                "semantics": semantics_json,
                "source": "offline-fallback",
                "ai_used": False,
                "note": f"AI error: {msg[:160]} (offline narration substituted)"
            })
    except Exception as e:
        return JSONResponse({"success": False, "error": f"Processing failed: {e}"}, status_code=500)
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        csv_path2 = temp_path + ".csv"
        if os.path.exists(csv_path2):
            try:
                os.remove(csv_path2)
            except Exception:
                pass
