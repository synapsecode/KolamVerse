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
from kolamgen_web import draw_kolam_web_bytes
from kolamspline import get_spline_csv, get_spline_json
from utils import load_ai_prompt_template
import io
import cv2
import numpy as np
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

app = FastAPI()
kolam_frame_manager = KolamFrameManager()

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/assets", StaticFiles(directory="assets"), name="assets")


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
async def upload_kolam(
    file: UploadFile = File(...),
    maxsize: int = Query(0, ge=0, description="Downscale longest side before tracing (0=disable)")
):
    # Check content type
    if not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Only image files are allowed"}, status_code=400)

    await kolam_frame_manager.clear()

    # Save uploaded file
    file_path = os.path.join(STATIC_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Optional downscale for speed on large images
    if maxsize and maxsize > 0:
        try:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                m = max(h, w)
                if m > maxsize:
                    scale = maxsize / float(m)
                    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(file_path, resized)
        except Exception:
            pass

    # Convert Image to CSV
    csv_filename = f"{os.path.splitext(file.filename)[0]}.csv"
    csv_path = os.path.join(STATIC_DIR, csv_filename)
    image_to_kolam_csv(file_path, csv_path)

    # Delete the Image
    os.remove(file_path)

    # TODO: REMOVE CSV

    return {"csv_file": csv_filename}


@app.get("/animate_kolam")
async def animate_kolam(csv_file: str = Query(..., description="CSV filename generated from /upload_kolam")):
    csv_path = os.path.join(STATIC_DIR, csv_file)

    if not os.path.exists(csv_path):
        return JSONResponse({"error": "CSV file not found"}, status_code=404)
    
    await kolam_frame_manager.clear()

    strokes = load_all_points(csv_path)
    strokes = normalize_strokes(strokes)
    path = compute_eulerian_path(strokes, tol=1e-1)

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

# ---------------- KolamSpline -------------------


@app.get("/kolamspline", response_class=HTMLResponse)
async def index(data: str=None):
    index_path = os.path.join(STATIC_DIR, "spline.html")
    return FileResponse(index_path)

@app.get("/spline_json")
def spline_json(
    csv_file: str = Query(..., description="CSV filename from /upload_kolam"),
    smooth: float = Query(0.0, ge=0.0)
):
    """Return B-spline parameters t (knots), c (coefficients), k (degree)."""
    csv_path = os.path.join(STATIC_DIR, csv_file)
    if not os.path.exists(csv_path):
        return JSONResponse({"error": "CSV file not found"}, status_code=404)

    res, err = get_spline_json(csv_path, smooth)
    if(err != None):
        return JSONResponse({"error": str(err)}, status_code=400)
    
    return JSONResponse(res)

@app.get("/spline_points")
def spline_points(
    csv_file: str = Query(..., description="CSV filename from /upload_kolam"),
    samples: int = Query(1000, ge=10, le=50000),
    smooth: float = Query(0.0, ge=0.0)
):
    csv_path = os.path.join(STATIC_DIR, csv_file)
    if not os.path.exists(csv_path):
        return JSONResponse({"error": "CSV file not found"}, status_code=404)
    data, err = get_spline_csv(csv_path, samples, smooth)
    if(err != None):
        return JSONResponse({"error": str(err)}, status_code=400)

    return Response(content=data, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=kolam_points.csv"
    })

# ---------------- KolamGen -------------------

@app.get("/kolamgen", response_class=HTMLResponse)
def app_kolamgen():
    index_path = os.path.join(STATIC_DIR, "kolamgen.html")
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
    
    temp_path = os.path.join(STATIC_DIR, f"ai_{file.filename}")
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        print("ERRRR",e)
        return JSONResponse({"success": False, 'error': 'file copy error'})

    # ------- Remove Eventually --------
    csv_path = temp_path + ".csv"
    image_to_kolam_csv(temp_path, csv_path)
    strokes = load_all_points(csv_path)
    strokes = normalize_strokes(strokes)
    eulerian_path = compute_eulerian_path(strokes, tol=1e-1)

    resp, err = await describe_kolam_using_ai(
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

# --------- Kolam Playground ----------------

@app.get("/kolamplayground", response_class=HTMLResponse)
def playground():
    index_path = os.path.join(STATIC_DIR, "kolamplayground.html")
    return FileResponse(index_path)

# ---------- Practice Kolam Section -------------------

@app.get("/practice", response_class=HTMLResponse)
def playground(data: str = None):
    index_path = os.path.join(STATIC_DIR, "practicekolam.html")
    return FileResponse(index_path)
