# KolamCreate

SIH 25107 - Develop computer programs (in any language, preferably Python) to identify the design principles behind the Kolam designs and recreate the kolams.

Kolams (known by other narnes as muggu, rangoli and rangavalli) are significant cultural traditions of India, blending art, ingenuity, and culture.

The designs vary by region, and the designs consist of grids of dots, with symmetry, repetition, and spatial reasoning embedded in them.

The Kolam designs provide a fascinating area of study for their strong mathematical underpinnings.

The challenge is to develop computer programs (in any language, preferably Python) to identify the design principles behind the Kolam designs and recreate the kolams.


## Kolam Rules
- Rule 1: Uniformly spacing of dots
- Rule 2: Smooth drawing line around the dots
- Rule 3: Symmetry in drawings
- Rule 4: Straight lines are drawn inclined at an angle of 45 degrees
- Rule 5: Drawing lines never trace back
- Rule 6: Arcs adjoining the dots
- Rule 7 : Kolam is completed when all points are enclosed by the drawing line

## Execution
1. Create venv (first time)
	- Windows PowerShell: `python -m venv venv; .\venv\Scripts\Activate.ps1`
	- macOS/Linux: `python3 -m venv venv; source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (optional; without it AI narration falls back to offline narration)
4. Run server (development): `uvicorn main:app --reload --port 8000`
5. Open:
	- KolamGen:  http://127.0.0.1:8000/kolamgen
	- KolamTrace: http://127.0.0.1:8000/kolamtrace

If `PyTurboJPEG` native library is missing the app transparently falls back to OpenCV JPEG encoding.

### Environment Variables
| Variable | Required | Purpose |
|----------|----------|---------|
| GEMINI_API_KEY | No | Enables Gemini-based AI narration & prompt→seed generation. Without it offline narration is used. |

`GEMINI_API_KEY` is read from `.env` (UTF-8, no BOM). Never commit the real key.

### Typical Workflow
1. Modify code / HTML in `static/` or JS in `assets/`.
2. Run locally & verify both pages.
3. Commit changes on a feature branch: `git add . && git commit -m "feat: ..."`
4. Push & open a Pull Request (PR) against the main branch.

### PR Checklist
- [ ] No merge conflict markers remain
- [ ] `.env` NOT committed (verify with `git status`)
- [ ] New assets referenced correctly (cache-busting if needed)
- [ ] Manual smoke test: upload image, view animation, description + narration
- [ ] Seed generation & narration working on KolamGen page

---


Deprecated (Inside legacy_code)
3. To Run the Turtle Version: python kolam_turtle.py
4. To Run the Manim Version: manim -pql kolam_manim.py Kolam
5. To Run the Kolam2CSV part: python kolam2csv.py --img ./sample1.jpeg --csv ./ab.csv
6. To Run the Kolam Tracer: python kolamtrace.py --csv "./abc1.csv"

Review Tips:

2.⁠ ⁠Make a Live Simulator along with the GenAI: needs to allow control of repititions mirrors etc
3.⁠ ⁠⁠implement the image to natural language (user pus kolam and they get a design description)
5.⁠ ⁠⁠implement the splines part into the webview itself and dont use dots tery to make those dots into a mathematical curve
8.⁠ ⁠⁠add more screenshorts and links to ppt

## Security / Secrets
- Do not commit `.env` (listed in `.gitignore`).
- Use `.env.example` for documentation of required variables.

## Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| Raw CSS/JS showing at top of page | Malformed `<head>` or stray style outside `<head>` | Fixed; ensure only one `<head>` tag and externalize scripts |
| `RuntimeError: Unable to locate turbojpeg library` | Missing native libjpeg-turbo | Fallback now uses OpenCV; optionally install system libjpeg-turbo for performance |
| BOM / UnicodeDecodeError when loading `.env` | File saved with UTF-16 or BOM | Recreate `.env` in UTF-8 without BOM; helper strips BOM defensively |
| AI narration always offline | Missing or invalid `GEMINI_API_KEY` | Set valid key in `.env` and restart server |
