import json
import math
from typing import List, Tuple, Dict, Any

import numpy as np

Point = Tuple[float, float]
Edge = Tuple[Point, Point]


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != (2,) or b.shape != (2,):
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    cosv = float(np.dot(a, b) / (na * nb))
    cosv = max(-1.0, min(1.0, cosv))
    ang = math.degrees(math.acos(cosv))
    cross = a[0] * b[1] - a[1] * b[0]
    return ang if cross >= 0 else -ang


def _detect_loops(points: np.ndarray, tol: float = 5.0) -> int:
    # Simple loop count: count returns near start excluding trivial first point
    loops = 0
    start = points[0]
    for i in range(10, len(points)):
        if np.linalg.norm(points[i] - start) < tol:
            loops += 1
            # move start forward to avoid recounting same closure
            start = points[i]
    return loops


def build_semantic_representation(path: List[Edge], max_steps: int = 250) -> Dict[str, Any]:
    """Convert an Eulerian path (list of edges) into a compact semantic JSON-ready dict."""
    if not path:
        return {"metadata": {"total_segments": 0}, "steps": []}

    # Flatten points from edges preserving order
    pts: List[Point] = [path[0][0]] + [v for (_, v) in path]
    P = np.array(pts, dtype=float)

    # Normalize into [0,1]
    min_xy = P.min(axis=0)
    max_xy = P.max(axis=0)
    span = max_xy - min_xy
    span[span == 0] = 1.0
    N = (P - min_xy) / span

    # Compute steps
    steps = []
    for i in range(1, len(N)):
        prev = N[i - 1]
        cur = N[i]
        vec = cur - prev
        seg_len = float(np.linalg.norm(vec))
        if i >= 2:
            prev_vec = prev - N[i - 2]
            turn = _angle_between(prev_vec, vec)
        else:
            turn = 0.0
        steps.append({
            "step": i,
            "from": [round(float(prev[0]), 4), round(float(prev[1]), 4)],
            "to": [round(float(cur[0]), 4), round(float(cur[1]), 4)],
            "len": round(seg_len, 4),
            "turn": round(turn, 2)
        })

    # Basic symmetry estimation (vertical/horizontal) by mirroring points
    vertical_sym = False
    horizontal_sym = False
    mirrored_x = np.copy(N)
    mirrored_x[:, 0] = 1 - mirrored_x[:, 0]
    mirrored_y = np.copy(N)
    mirrored_y[:, 1] = 1 - mirrored_y[:, 1]
    # use Hausdorff-like quick check via distance matrix mins
    def _approx_match(A, B):
        from scipy.spatial import cKDTree
        tree = cKDTree(B)
        d, _ = tree.query(A, k=1)
        return float(np.mean(d)) < 0.02  # tolerance

    try:
        vertical_sym = _approx_match(N, mirrored_x)
        horizontal_sym = _approx_match(N, mirrored_y)
    except Exception:
        pass

    loops = _detect_loops(P)

    # Trim steps for prompt efficiency
    trimmed = steps[:max_steps]
    trimmed_count = len(steps) - len(trimmed)

    meta = {
        "total_segments": len(steps),
        "trimmed_segments": trimmed_count if trimmed_count > 0 else 0,
        "loops_detected": loops,
        "estimated_symmetry": [s for s, flag in [("vertical", vertical_sym), ("horizontal", horizontal_sym)] if flag]
    }

    return {"metadata": meta, "steps": trimmed}


def semantic_json_string(path: List[Edge]) -> str:
    rep = build_semantic_representation(path)
    return json.dumps(rep, separators=(",", ":"))


__all__ = [
    "build_semantic_representation",
    "semantic_json_string"
]
