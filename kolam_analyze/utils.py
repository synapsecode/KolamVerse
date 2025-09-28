import base64

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