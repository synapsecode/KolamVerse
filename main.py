import base64
import os
import shutil
from fastapi import FastAPI, File, Response, UploadFile, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from kolam2csv import image_to_kolam_csv
from kolam_frame_manger import KolamFrameManager
from kolamanimator import animate_eulerian_stream, compute_eulerian_path, load_all_points, normalize_strokes
from kolamdrawv2 import draw_kolam_from_seed

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
        animate_eulerian_stream(path, kolam_frame_manager, step_delay=0.005),
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

# ---------------- KolamDraw -------------------

@app.get("/kolamdraw", response_class=HTMLResponse)
def app_kolamdraw():
    index_path = os.path.join(STATIC_DIR, "kolamdraw.html")
    return FileResponse(index_path)

@app.get("/drawkolam")
def drawkolam(seed: str = "FBFBFBFB", depth: int = 1):
    img_bytes = draw_kolam_from_seed(seed=seed, depth=depth)
    return Response(content=img_bytes, media_type="image/png")