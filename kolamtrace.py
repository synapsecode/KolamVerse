import argparse
import time
import numpy as np
import pandas as pd
import turtle as T
import networkx as nx

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

def animate_eulerian(path, step_delay=0.01, bg="#2e5f3b"):
    screen = T.Screen()
    screen.setup(width=900, height=900)
    screen.bgcolor(bg)
    pen = T.Turtle(visible=False)
    pen.color("white")
    pen.pensize(3)
    pen.speed(0)
    T.tracer(0, 0)

    # Move to start
    start = path[0][0]
    pen.up()
    pen.goto(*start)
    pen.down()

    for (u, v) in path:
        pen.goto(*v)
        T.update()
        if step_delay > 0:
            time.sleep(step_delay)

    T.update()
    T.done()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--delay", type=float, default=0)
    ap.add_argument("--tol", type=float, default=1e-1, help="Snapping tolerance")
    args = ap.parse_args()

    strokes = load_all_points(args.csv)
    strokes = normalize_strokes(strokes)

    path = compute_eulerian_path(strokes, tol=args.tol)
    animate_eulerian(path, step_delay=args.delay)
