from kolamanimator import load_all_points, normalize_strokes
import numpy as np
from scipy.interpolate import splprep, splev


def get_tck(csv_path, smooth):
    strokes = load_all_points(csv_path)
    if not strokes:
        return (None, 'NO_STROKES')
    strokes = normalize_strokes(strokes)
    P = np.vstack(strokes)
    diff = np.diff(P, axis=0)
    keep = np.ones(len(P), dtype=bool)
    keep[1:] = (np.linalg.norm(diff, axis=1) > 1e-6)
    P = P[keep]
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] == 0:
        return (None, 'DEGENERATE_PATH')
    u = d / d[-1]
    tck, _ = splprep([P[:, 0], P[:, 1]], u=u, s=smooth)
    return (tck, None)

def get_spline_json(csv_path, smooth):
    tck, err = get_tck(csv_path, smooth)
    if(err != None):
        return (None, err)
    t, c, k = tck
    cx, cy = c
    return ({
        "type": "bspline",
        "degree": int(k),
        "knots": t.tolist(),
        "cx": cx.tolist(),
        "cy": cy.tolist(),
        "notes": "Parametric B-spline: x(t)=sum_i cx[i]*B_{i,k}(t), y(t)=sum_i cy[i]*B_{i,k}(t). t in [knots[k], knots[-k-1]]"
    }, None)

def get_spline_csv(csv_path, samples, smooth):
    tck, err = get_tck(csv_path, smooth)
    if(err != None):
        return (None, err)
    u_new = np.linspace(0, 1, samples)
    x_new, y_new = splev(u_new, tck)
    C = np.column_stack([x_new, y_new])
    lines = ["x,y"] + [f"{p[0]},{p[1]}" for p in C]
    data = "\n".join(lines).encode()
    return (data, None)