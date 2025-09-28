from collections import Counter
from .offline_narrator import offline_narrate
from .kolam_semantics import semantic_json_string
from lsystem import generate_lsystem_state

def generate_seed_description(seed):   
    # Count character occurrences
    char_counts = Counter(seed)
    
    description = f"This L-system seed '{seed}' will generate a kolam pattern with the following characteristics: "
    
    # Analyze seed components
    if char_counts.get('F', 0) > 0:
        description += f"The pattern includes {char_counts['F']} forward line segment{'s' if char_counts['F'] > 1 else ''}. "
    
    if char_counts.get('A', 0) > 0:
        description += f"It contains {char_counts['A']} arc element{'s' if char_counts['A'] > 1 else ''} that create smooth curved sections. "
    
    if char_counts.get('B', 0) > 0:
        description += f"The design incorporates {char_counts['B']} complex arc motif{'s' if char_counts['B'] > 1 else ''} with intricate curved patterns. "
    
    if char_counts.get('L', 0) > 0 or char_counts.get('R', 0) > 0:
        turns = char_counts.get('L', 0) + char_counts.get('R', 0)
        description += f"The pattern includes {turns} directional change{'s' if turns > 1 else ''} that create angular variations. "
    
    # Analyze pattern characteristics
    seed_length = len(seed)
    if seed_length <= 4:
        description += "This is a simple pattern with minimal complexity. "
    elif seed_length <= 8:
        description += "This represents a moderately complex kolam design. "
    else:
        description += "This creates an intricate kolam with high complexity. "

    unique_chars = len(set(seed))
    if unique_chars < seed_length / 2:
        description += "The seed shows repetitive elements that will create rhythmic patterns in the final design. "

    if seed == seed[::-1]:
        description += "The seed is palindromic, which will result in symmetrical kolam patterns. "

    description += "When expanded through L-system iterations, this seed will generate a traditional kolam following the fundamental rules of continuous line drawing around dots."
    return description

def simulate_kolam_path(seed: str = "FBFBFBFB", depth: int = 1, step: int = 20, angle: int = 0):
    """Lightweight path simulation for seeds (used only for offline seed narration)."""
    from math import cos, sin, radians, sqrt
    state = generate_lsystem_state(seed, depth)
    x, y = 0.0, 0.0
    heading = float(angle)
    path_points = [(x, y)]

    def add_arc(x, y, heading, radius, extent_deg, steps=20):
        points = []
        step_angle = radians(extent_deg / steps)
        cur_x, cur_y = x, y
        for i in range(steps):
            theta1 = radians(heading) + i * step_angle
            theta2 = radians(heading) + (i + 1) * step_angle
            x_next = cur_x + radius * (sin(theta2) - sin(theta1))
            y_next = cur_y - radius * (cos(theta2) - cos(theta1))
            points.append((x_next, y_next))
            cur_x, cur_y = x_next, y_next
        heading += extent_deg
        return points, heading, (cur_x, cur_y)

    for ch in state:
        if ch == 'F':
            x2 = x + step * cos(radians(heading))
            y2 = y + step * sin(radians(heading))
            path_points.append((x2, y2))
            x, y = x2, y2
        elif ch == 'A':
            arc_points, heading, (x, y) = add_arc(x, y, heading, radius=step * 1.2, extent_deg=90)
            path_points.extend(arc_points)
        elif ch == 'B':
            I = step / sqrt(2)
            x2 = x + I * cos(radians(heading))
            y2 = y + I * sin(radians(heading))
            path_points.append((x2, y2))
            x, y = x2, y2
            arc_points, heading, (x, y) = add_arc(x, y, heading, radius=I * 1.1, extent_deg=270)
            path_points.extend(arc_points)
            x2 = x + I * cos(radians(heading))
            y2 = y + I * sin(radians(heading))
            path_points.append((x2, y2))
            x, y = x2, y2
        elif ch == 'L':
            heading += 45
        elif ch == 'R':
            heading -= 45
        else:
            pass
    # Convert to (point, point) edge list style elsewhere; return raw points here
    return path_points, []

def seed_narration(seed, depth):
    try:
        pts, types = simulate_kolam_path(seed=seed, depth=depth)
        if len(pts) < 2:
            return (None,"Seed produced insufficient points.")
        path_edges = list(zip(pts[:-1], pts[1:]))
        sem_json = semantic_json_string(path_edges)
        narration = offline_narrate(sem_json)
        return ({
            "success": True,
            "seed": seed,
            "depth": depth,
            "narration": narration,
            "source": "offline-seed"
        }, None)
    except Exception as e:
        return (None, str(e))
    