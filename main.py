import base64
import os
import shutil
from fastapi import FastAPI, File, Response, UploadFile, Query, Body
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from gem_config import configure_gemini
from kolam2csv import image_to_kolam_csv
from kolam_frame_manger import KolamFrameManager
from kolamanimator import animate_eulerian_stream, compute_eulerian_path, load_all_points, normalize_strokes
from kolamdraw_web import draw_kolam_web_bytes
from utils import load_ai_prompt_template
import io
import cv2
import numpy as np
from scipy.interpolate import splprep, splev

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
    return FileResponse(index_path)

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
    snapshots = await kolam_frame_manager.get_frames()
    
    if not snapshots:
        return JSONResponse({"error": "Snapshots not ready yet"}, status_code=404)

    frames_b64 = [base64.b64encode(f).decode("utf-8") for f in snapshots]

    return JSONResponse(
        {"frames": frames_b64},
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
    )

# ---------------- Spline Curve Render -------------------

def _hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (b, g, r)
    return (59, 95, 46)  # fallback to #2e5f3b


@app.get("/spline_curve")
def spline_curve(
    csv_file: str = Query(..., description="CSV filename from /upload_kolam"),
    width: int = Query(410, ge=100, le=2000),
    height: int = Query(370, ge=100, le=2000),
    samples: int = Query(1200, ge=100, le=20000),
    smooth: float = Query(0.0, ge=0.0),
    bg: str = Query("#2e5f3b"),
    thickness: int = Query(3, ge=1, le=12)
):
    """Render a smooth spline curve PNG from the uploaded CSV (no dots)."""
    csv_path = os.path.join(STATIC_DIR, csv_file)
    if not os.path.exists(csv_path):
        return JSONResponse({"error": "CSV file not found"}, status_code=404)

    # Load strokes and normalize to common scale like animator
    strokes = load_all_points(csv_path)
    if not strokes:
        return JSONResponse({"error": "No strokes found in CSV"}, status_code=400)
    strokes = normalize_strokes(strokes)

    # Concatenate into single polyline
    P = np.vstack(strokes)
    # Remove duplicates to avoid spline issues
    diff = np.diff(P, axis=0)
    keep = np.ones(len(P), dtype=bool)
    keep[1:] = (np.linalg.norm(diff, axis=1) > 1e-6)
    P = P[keep]

    # Fit parametric spline x(u), y(u)
    try:
        # Parameterize by cumulative distance for stability
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        if d[-1] == 0:
            raise ValueError("Degenerate path")
        u = d / d[-1]
        # splprep expects data as [x, y] with parameter u
        tck, _ = splprep([P[:, 0], P[:, 1]], u=u, s=smooth)
        u_new = np.linspace(0, 1, samples)
        x_new, y_new = splev(u_new, tck)
        C = np.column_stack([x_new, y_new])
    except Exception:
        # Fallback to original polyline
        C = P

    # Render to image using OpenCV
    canvas = np.full((height, width, 3), _hex_to_bgr(bg), dtype=np.uint8)

    # Compute transform to fit curve into image with padding
    pad = 0.05  # 5% border
    xmin, ymin = C.min(axis=0)
    xmax, ymax = C.max(axis=0)
    span_x = max(xmax - xmin, 1e-6)
    span_y = max(ymax - ymin, 1e-6)
    scale = min((width * (1 - 2 * pad)) / span_x, (height * (1 - 2 * pad)) / span_y)

    def to_px(pt):
        x = int((pt[0] - xmin) * scale + width * pad)
        y = int((pt[1] - ymin) * scale + height * pad)
        # Flip Y for image coordinates
        return (x, height - y)

    # Draw polyline (smooth look via dense sampling)
    for i in range(1, len(C)):
        cv2.line(canvas, to_px(C[i - 1]), to_px(C[i]), (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    ok, buf = cv2.imencode('.png', canvas)
    if not ok:
        return JSONResponse({"error": "Failed to encode image"}, status_code=500)
    return Response(content=buf.tobytes(), media_type="image/png")


@app.get("/spline_json")
def spline_json(
    csv_file: str = Query(..., description="CSV filename from /upload_kolam"),
    smooth: float = Query(0.0, ge=0.0)
):
    """Return B-spline parameters t (knots), c (coefficients), k (degree)."""
    csv_path = os.path.join(STATIC_DIR, csv_file)
    if not os.path.exists(csv_path):
        return JSONResponse({"error": "CSV file not found"}, status_code=404)

    strokes = load_all_points(csv_path)
    if not strokes:
        return JSONResponse({"error": "No strokes found in CSV"}, status_code=400)
    strokes = normalize_strokes(strokes)
    P = np.vstack(strokes)
    diff = np.diff(P, axis=0)
    keep = np.ones(len(P), dtype=bool)
    keep[1:] = (np.linalg.norm(diff, axis=1) > 1e-6)
    P = P[keep]

    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] == 0:
        return JSONResponse({"error": "Degenerate path"}, status_code=400)
    u = d / d[-1]

    tck, _ = splprep([P[:, 0], P[:, 1]], u=u, s=smooth)
    t, c, k = tck
    cx, cy = c
    return JSONResponse({
        "type": "bspline",
        "degree": int(k),
        "knots": t.tolist(),
        "cx": cx.tolist(),
        "cy": cy.tolist(),
        "notes": "Parametric B-spline: x(t)=sum_i cx[i]*B_{i,k}(t), y(t)=sum_i cy[i]*B_{i,k}(t). t in [knots[k], knots[-k-1]]"
    })


@app.get("/spline_points")
def spline_points(
    csv_file: str = Query(..., description="CSV filename from /upload_kolam"),
    samples: int = Query(1000, ge=10, le=50000),
    smooth: float = Query(0.0, ge=0.0)
):
    """Return sampled points along the spline as CSV (x,y)."""
    csv_path = os.path.join(STATIC_DIR, csv_file)
    if not os.path.exists(csv_path):
        return JSONResponse({"error": "CSV file not found"}, status_code=404)

    strokes = load_all_points(csv_path)
    if not strokes:
        return JSONResponse({"error": "No strokes found in CSV"}, status_code=400)
    strokes = normalize_strokes(strokes)
    P = np.vstack(strokes)
    diff = np.diff(P, axis=0)
    keep = np.ones(len(P), dtype=bool)
    keep[1:] = (np.linalg.norm(diff, axis=1) > 1e-6)
    P = P[keep]

    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] == 0:
        return JSONResponse({"error": "Degenerate path"}, status_code=400)
    u = d / d[-1]

    tck, _ = splprep([P[:, 0], P[:, 1]], u=u, s=smooth)
    u_new = np.linspace(0, 1, samples)
    x_new, y_new = splev(u_new, tck)
    C = np.column_stack([x_new, y_new])

    # Build CSV bytes
    lines = ["x,y"] + [f"{p[0]},{p[1]}" for p in C]
    data = "\n".join(lines).encode()
    return Response(content=data, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=kolam_points.csv"
    })


# ---------------- KolamDraw -------------------

@app.get("/kolamdraw", response_class=HTMLResponse)
def app_kolamdraw():
    index_path = os.path.join(STATIC_DIR, "kolamdraw.html")
    return FileResponse(index_path)

@app.get("/drawkolam")
def drawkolam(seed: str = "FBFBFBFB", depth: int = 1):
    # Use the web-compatible turtle implementation that matches kolamdraw.py exactly
    # Color mode is controlled by 'C' commands in the seed string
    img_bytes = draw_kolam_web_bytes(seed=seed, depth=depth)
    return Response(content=img_bytes, media_type="image/png")

@app.post("/generate_seed_from_prompt")
async def generate_seed_from_prompt(payload: dict = Body(...)):
    user_prompt = payload.get("prompt")
    if not user_prompt:
        return JSONResponse({"error": "Prompt cannot be empty"}, status_code=400)

    # Load the instructional prompt from external file
    prompt_template = load_ai_prompt_template()
    instructional_prompt = prompt_template.format(user_prompt=user_prompt)
    try:
        response = configure_gemini().generate_content(instructional_prompt)
        generated_seed = response.text.strip()
        
        # Basic validation to ensure it only contains allowed characters
        if all(c in "FABLRC" for c in generated_seed):
             return JSONResponse({"seed": generated_seed})
        else:
             # Fallback or error if the model returns invalid text
             return JSONResponse({"error": "Failed to generate a valid seed.", "details": generated_seed}, status_code=500)

    except Exception as e:
        return JSONResponse({"error": "An error occurred with the AI model.", "details": str(e)}, status_code=500)

