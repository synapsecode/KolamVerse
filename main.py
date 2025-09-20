import os
import shutil
from fastapi import FastAPI, File, Response, UploadFile, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from kolam2csv import image_to_kolam_csv
from kolamanimator import animate_eulerian_stream, compute_eulerian_path, load_all_points, normalize_strokes
from kolamdrawv2 import draw_kolam_from_seed

app = FastAPI()

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)


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
def upload_kolam(file: UploadFile = File(...)):
    # Check content type
    if not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Only image files are allowed"}, status_code=400)

    # Save uploaded file
    file_path = os.path.join(STATIC_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Convert Image to CSV
    csv_filename = f"{os.path.splitext(file.filename)[0]}.csv"
    csv_path = os.path.join(STATIC_DIR, csv_filename)
    image_to_kolam_csv(file_path, csv_path)

    return {"csv_file": csv_filename}


@app.get("/animate_kolam")
async def animate_kolam(csv_file: str = Query(..., description="CSV filename generated from /upload_kolam")):
    csv_path = os.path.join(STATIC_DIR, csv_file)

    if not os.path.exists(csv_path):
        return JSONResponse({"error": "CSV file not found"}, status_code=404)

    strokes = load_all_points(csv_path)
    strokes = normalize_strokes(strokes)
    path = compute_eulerian_path(strokes, tol=1e-1)

    return StreamingResponse(
        animate_eulerian_stream(path, step_delay=0.005),
        media_type="multipart/x-mixed-replace; boundary=frame"
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