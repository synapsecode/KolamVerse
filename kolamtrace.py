import argparse
import time
import numpy as np
import pandas as pd
import turtle as T

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

def segments_intersect(p1, p2, q1, q2):
    # Check if two segments (p1-p2 and q1-q2) intersect
    def ccw(a, b, c):
        return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

def animate(strokes, step_delay=0.01, stride=1, bg="#2e5f3b"):
    screen = T.Screen()
    screen.setup(width=900, height=900)
    screen.bgcolor(bg)
    pen = T.Turtle(visible=False)
    pen.color("white")
    pen.pensize(3)
    pen.speed(0)
    T.tracer(0, 0)

    drawn_segments = []  # List of all previously drawn segments

    for P in strokes:
        pen.up()
        start = tuple(P[0])
        pen.goto(*start)
        pen.down()

        for i in range(1, len(P)):
            end = tuple(P[i])
            overlap = False

            # Check intersection with all previously drawn segments
            for seg_start, seg_end in drawn_segments:
                if segments_intersect(start, end, seg_start, seg_end):
                    overlap = True
                    break

            if not overlap:
                pen.goto(*end)
                drawn_segments.append((start, end))
                if i % stride == 0:
                    T.update()
                    if step_delay > 0:
                        time.sleep(step_delay)
                start = end
            else:
                # Skip drawing this segment, lift the pen
                pen.up()
                pen.goto(*end)
                pen.down()
                start = end

    T.update()
    T.done()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--delay", type=float, default=0)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    strokes = load_all_points(args.csv)
    strokes = normalize_strokes(strokes)
    animate(strokes, step_delay=args.delay, stride=args.stride)
