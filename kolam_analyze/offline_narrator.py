"""Heuristic offline narration builder for kolam drawing steps.

Takes the semantic JSON produced by `kolam_semantics.semantic_json_string` and
produces a structured, human‑readable narration describing the drawing phases.
"""
from __future__ import annotations

import json
import math
from typing import Dict, Any, List


def _direction_category(dx: float, dy: float) -> str:
    """Classify approximate direction for descriptive vocabulary."""
    ang = math.degrees(math.atan2(dy, dx))  # -180..180
    # Normalize to 0..360
    if ang < 0:
        ang += 360
    # Cardinal / diagonal buckets
    buckets = [
        ("east", 0),
        ("northeast", 45),
        ("north", 90),
        ("northwest", 135),
        ("west", 180),
        ("southwest", 225),
        ("south", 270),
        ("southeast", 315)
    ]
    best = min(buckets, key=lambda b: abs(b[1] - ang))
    return best[0]


def _cluster_arc_sequences(turns: List[float], turn_indices: List[int]) -> int:
    """Count sequences of consecutive gentle arcs to estimate 'petal' motif groups."""
    if not turn_indices:
        return 0
    clusters = 1
    last_idx = turn_indices[0]
    for idx in turn_indices[1:]:
        if idx != last_idx + 1:  # gap -> new cluster
            clusters += 1
        last_idx = idx
    return clusters


def offline_narrate(semantics_json: str) -> str:
    try:
        data: Dict[str, Any] = json.loads(semantics_json)
    except Exception:
        return "Kolam structure detected, but detailed semantic decoding was unavailable."  # fallback

    steps: List[Dict[str, Any]] = data.get("steps", [])
    meta = data.get("metadata", {})
    total = len(steps)
    if total == 0:
        return "An empty or non‑traceable pattern was provided."

    # Partition indices for phases
    p1_end = max(1, int(total * 0.2))
    p3_start = max(p1_end + 1, int(total * 0.8))

    def slice_stats(start: int, end: int):
        segs = steps[start:end]
        straight = 0
        gentle = 0
        sharp = 0
        directions = []
        arc_turn_indices = []
        for i, s in enumerate(segs, start=start):
            turn = abs(s.get("turn", 0.0))
            dx = s.get("to", [0, 0])[0] - s.get("from", [0, 0])[0]
            dy = s.get("to", [0, 0])[1] - s.get("from", [0, 0])[1]
            directions.append(_direction_category(dx, dy))
            if turn <= 10:
                straight += 1
            elif turn <= 50:
                gentle += 1
                arc_turn_indices.append(i)
            else:
                sharp += 1
        total_local = len(segs) or 1
        return {
            "straight_ratio": straight / total_local,
            "gentle_ratio": gentle / total_local,
            "sharp_ratio": sharp / total_local,
            "dominant_dir": max(set(directions), key=directions.count) if directions else "east",
            "gentle_cluster_count": _cluster_arc_sequences([s.get("turn", 0.0) for s in segs], arc_turn_indices)
        }

    stats_foundation = slice_stats(0, p1_end)
    stats_body = slice_stats(p1_end, p3_start)
    stats_closure = slice_stats(p3_start, total)

    symmetry = meta.get("estimated_symmetry", [])
    loops = meta.get("loops_detected", 0)
    trimmed = meta.get("trimmed_segments", 0)
    sampled_from = meta.get("sampled_from")

    # Motif inference heuristics
    petals = False
    if 3 <= stats_body["gentle_cluster_count"] <= 8 and stats_body["gentle_ratio"] > 0.25:
        petals = True

    lattice = stats_foundation["straight_ratio"] > 0.55 and stats_body["straight_ratio"] > 0.35

    parts: List[str] = []
    overview_bits = []
    if symmetry:
        overview_bits.append(
            "symmetric (" + ", ".join(symmetry) + ")"
        )
    if lattice:
        overview_bits.append("built over a straight/diagonal lattice foundation")
    if petals:
        overview_bits.append("adorned with repeating arc petals")
    if loops:
        overview_bits.append(f"containing {loops} loop closure{'s' if loops!=1 else ''}")

    overview = "This kolam is " + (", ".join(overview_bits) if overview_bits else "a continuous line design") + "."
    parts.append(overview)

    # Phase 1
    if lattice:
        p1_desc = (
            "Foundation Phase: The drawing begins by laying out a lattice of "
            f"predominantly straight or gently aligned segments heading mainly {stats_foundation['dominant_dir']}."
        )
    else:
        p1_desc = (
            "Foundation Phase: The initial strokes establish the central core with a mixture of minor turns "
            f"and direction shifts oriented toward {stats_foundation['dominant_dir']}."
        )
    parts.append(p1_desc)

    # Phase 2
    if petals:
        p2_desc = (
            "Expansion Phase: Repeating gentle arc groups radiate outward forming petal-like enclosures that "
            "mirror across the detected symmetry axes."
        )
    else:
        if stats_body["gentle_ratio"] > stats_body["straight_ratio"]:
            p2_desc = (
                "Expansion Phase: The middle section favors flowing curved turns that weave around earlier lines, "
                "maintaining continuity without retracing."
            )
        else:
            p2_desc = (
                "Expansion Phase: Structured straight links extend the mesh outward, reinforcing rhythmic repetition."
            )
    parts.append(p2_desc)

    # Phase 3
    closure_clause = []
    if loops:
        closure_clause.append("closing loops")
    if stats_closure["gentle_ratio"] > 0.3:
        closure_clause.append("final soft arcs")
    else:
        closure_clause.append("tight linking strokes")
    p3_desc = (
        "Closure Phase: The path finishes with " + ", and ".join(closure_clause) +
        ", sealing remaining gaps and returning near its origin without overlap."
    )
    parts.append(p3_desc)

    if trimmed:
        parts.append(
            "(Note: Long path was summarized; some micro-turns omitted for narration clarity.)"
        )
    if sampled_from:
        parts.append(f"Analyzed subset of {meta.get('kept_steps', 'N')} steps sampled from {sampled_from} total segments.")

    parts.append(
        "Overall, the continuous line respects kolam principles: flow, enclosure of space, and balanced repetition."
    )

    return " ".join(parts)


__all__ = ["offline_narrate"]
