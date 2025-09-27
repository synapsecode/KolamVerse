import base64
import os
import shutil
from fastapi import FastAPI, File, Response, UploadFile, Query, Body
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from kolam2csv import image_to_kolam_csv
from kolam_frame_manger import KolamFrameManager
from kolamanimator import animate_eulerian_stream, compute_eulerian_path, load_all_points, normalize_strokes
from kolamdrawv2 import draw_kolam_from_seed
from kolamdraw import draw_kolam
from kolamdraw_web import draw_kolam_web_bytes
import google.generativeai as genai

genai.configure(api_key="XXXXXXXXXXXXX")
model = genai.GenerativeModel('gemini-2.5-pro')

app = FastAPI()
kolam_frame_manager = KolamFrameManager()

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
    # Use the web-compatible turtle implementation that matches kolamdraw.py exactly
    # Color mode is controlled by 'C' commands in the seed string
    img_bytes = draw_kolam_web_bytes(seed=seed, depth=depth)
    return Response(content=img_bytes, media_type="image/png")


@app.post("/generate_seed_from_prompt")
async def generate_seed_from_prompt(payload: dict = Body(...)):
    user_prompt = payload.get("prompt")
    if not user_prompt:
        return JSONResponse({"error": "Prompt cannot be empty"}, status_code=400)

    # This is the master prompt that instructs the AI
    instructional_prompt = f"""
You are an expert designer of Kolam art using a specific L-system that grows recursively. Your task is to convert a user's description into a simple starting seed string, called an axiom. The system will then expand this axiom to create the full, intricate pattern.

The available alphabet and their drawing actions are:
- 'F': Draws a straight line. It does not change or grow in recursions.
- 'A': Places a dot, then draws a 90-degree arc. In the next iteration, 'A' will be replaced by 'AFBFA'.
- 'B': Places a dot, then draws a decorative 'petal' shape (a 270-degree loop). In the next iteration, 'B' will be replaced by 'AFBFBFBFA'.
- 'C': Toggles between colorful mode (green/blue/red lines with black dots) and monochrome mode (white lines and dots). Does not change in recursions.

CRITICAL RULES:
1.  **Do NOT use 'L' or 'R' commands.** There are no explicit turns in this system. All turns are part of the 'A' and 'B' shapes.
2.  Your goal is to create a simple, symmetrical starting seed (axiom). The complexity will come from the L-system's expansion, not from a long seed.
3.  The axiom should be a repeating pattern that forms a closed loop, like `FBFBFB` or `ABABAB`.
4.  Use the 'C' command strategically to control which parts of the design will be colorful.

Here are some examples of converting a description to a starting seed (axiom):

- Description: "A simple square-like shape made of straight lines and petals."
  Axiom: FBFBFB

- Description: "A design that starts with rounded corners."
  Axiom: AAAA

- Description: "A four-petaled flower shape that will grow more complex."
  Axiom: BBBB

- Description: "An alternating pattern of straight lines and rounded corners."
  Axiom: AFAFAF

When using C:

- Description: "A four-petaled flower shape that will grow more complex." #no color mentioned
    Axiom: BBBB
  
- Description: "A completely colorful four-petaled flower."
  Axiom: CBBBB

- Description: "A design with only colorful, rounded corners, and white connecting lines."
  Axiom: FCACFCAC

- Description: "A pattern that alternates between colorful petals and white lines."
  Axiom: CAFCAFCAFCAF

Now, convert the following user description into a simple L-system axiom. Only output the final axiom string and nothing else.

User Description: "{user_prompt}"
Axiom:
"""
    try:
        response = model.generate_content(instructional_prompt)
        generated_seed = response.text.strip()
        
        # Basic validation to ensure it only contains allowed characters
        if all(c in "FABLRC" for c in generated_seed):
             return JSONResponse({"seed": generated_seed})
        else:
             # Fallback or error if the model returns invalid text
             return JSONResponse({"error": "Failed to generate a valid seed.", "details": generated_seed}, status_code=500)

    except Exception as e:
        return JSONResponse({"error": "An error occurred with the AI model.", "details": str(e)}, status_code=500)

