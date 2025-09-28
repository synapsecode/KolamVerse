import base64
import os
import shutil
from typing import Dict
from fastapi import FastAPI, File, Response, UploadFile, Query, Body
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from gem_config import configure_gemini
from kolam2csv import image_to_kolam_csv
from kolam_analyze.kolamdescribe import describe_kolam_characteristics, describe_kolam_using_ai
from kolam_analyze.seed_desc import seed_narration
from kolam_frame_manager import KolamFrameManager
from kolamanimator import animate_eulerian_stream, compute_eulerian_path, load_all_points, normalize_strokes
from kolamdraw_web import draw_kolam_web_bytes
from utils import load_ai_prompt_template
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
    
# ---------------- Kolam Analysis & Description Section -------------------

@app.get("/narrate_seed")
def narrate_seed(seed: str = "FBFBFBFB", depth: int = 1):
    narration, err = seed_narration(seed, depth)
    if(err != None):
         return JSONResponse({"success": False, "error": str(err)}, status_code=500)
    return JSONResponse(narration)


@app.post("/describe_kolam")
async def describe_kolam(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Only image files are allowed", "success": False}, status_code=400)

    # Save uploaded file temporarily
    temp_filename = f"temp_{file.filename}"
    file_path = os.path.join(STATIC_DIR, temp_filename)

    characteristics, err = describe_kolam_characteristics(file.file, file_path)

    if(characteristics is None):
        return JSONResponse({
            "error": f"Analysis failed: {err}",
            "success": False
        }, status_code=500)
    else:
        return JSONResponse({
            "description": characteristics["description"],
            "features": characteristics["features"],
            "success": True
        })

@app.post("/describe_kolam_ai")
async def describe_kolam_ai(
    file: UploadFile = File(...),
    include_image: bool = True,
    compress: bool = True,
    mode: str = "auto"  # auto|offline|ai
):
    if not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Only image files are allowed", "success": False}, status_code=400)
    
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        print(e)
        return JSONResponse({"success": False, 'error': 'file copy error'})

    # ------- Remove Eventually --------
    temp_path = os.path.join(STATIC_DIR, f"ai_{file.filename}")
    csv_path = temp_path + ".csv"
    image_to_kolam_csv(temp_path, csv_path)
    strokes = load_all_points(csv_path)
    strokes = normalize_strokes(strokes)
    eulerian_path = compute_eulerian_path(strokes, tol=1e-1)

    resp, err = describe_kolam_using_ai(
        file,
        eulerian_path,
        include_image, 
        compress,
        mode,
    )
    if(err != None):
        return JSONResponse({"success": False, "error": str(err)})

    # Cleanup
    for fpath in [temp_path, csv_path]:
        try: os.remove(fpath)
        except Exception: pass

    return JSONResponse(resp)