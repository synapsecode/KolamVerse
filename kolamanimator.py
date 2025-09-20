import asyncio
import io
import cv2
import numpy as np
import pandas as pd
import turtle as T
import networkx as nx
from PIL import Image

def load_all_points(csv_path: str):
    df = pd.read_csv(csv_path)
    strokes = []
    for col in df.columns:
        if col.startswith("x-kolam"):
            idx = col.split()[-1]
            x = df[f"x-kolam {idx}"].to_numpy()
            y = df[f"y-kolam {idx}"].to_numpy()
            P = np.c_[x, y]
            P = P[np.isfinite(P).all(axis=1)]
            if len(P) < 2:
                continue
            strokes.append(P)
    return strokes

def normalize_strokes(strokes):
    all_pts = np.vstack(strokes)
    all_pts = all_pts - all_pts.mean(axis=0)
    span = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1]))
    span = span if span > 0 else 1.0
    all_pts = all_pts / span * 300.0
    result = []
    offset = 0
    for P in strokes:
        n = len(P)
        result.append(all_pts[offset:offset+n])
        offset += n
    return result

def snap_point(pt, tol=1e-1):
    """Round point to nearest grid to avoid floating mismatch."""
    return (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)

def build_graph(strokes, tol=1e-1):
    G = nx.MultiGraph()
    for P in strokes:
        for i in range(len(P)-1):
            u = snap_point(P[i], tol)
            v = snap_point(P[i+1], tol)
            G.add_edge(u, v)
    return G

def compute_eulerian_path(strokes, tol=1e-1):
    G = build_graph(strokes, tol)
    if nx.is_eulerian(G):
        path = list(nx.eulerian_circuit(G))
    elif nx.has_eulerian_path(G):
        path = list(nx.eulerian_path(G))
    else:
        print("Graph degrees:", dict(G.degree()))
        raise ValueError("No Eulerian path or circuit possible (even after snapping)")
    return path


# ------------------------ Turtle Part --------------------------------

async def animate_eulerian_stream(path, step_delay=0.05, bg=(46, 95, 59)):
    """Async MJPEG streamer for Eulerian path."""
    size = 900
    canvas = np.full((size, size, 3), bg, dtype=np.uint8)

    # Normalize coords
    xs = [p[0] for edge in path for p in edge]
    ys = [p[1] for edge in path for p in edge]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale = min(size / (max_x - min_x + 1), size / (max_y - min_y + 1)) * 0.9

    def to_px(pt):
        x = int((pt[0] - min_x) * scale + size * 0.05)
        y = int((pt[1] - min_y) * scale + size * 0.05)
        return (x, size - y)

    for (u, v) in path:
        cv2.line(canvas, to_px(u), to_px(v), (255, 255, 255), 3)

        _, buffer = cv2.imencode(".jpg", canvas)
        frame = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

        if step_delay > 0:
            await asyncio.sleep(step_delay)

    # Final frame
    _, buffer = cv2.imencode(".jpg", canvas)
    yield (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
    )