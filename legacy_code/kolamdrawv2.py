# import io
# from math import cos, sin, radians, sqrt
# from PIL import Image, ImageDraw
# from lsystem import generate_lsystem_state

# def simulate_kolam_path(seed="FBFBFBFB", depth=1, step=20, angle=0):
#     """
#     Simulate Kolam path with natural turtle-like arcs (no dots)
#     Returns path points and segment types
#     """
#     state = generate_lsystem_state(seed, depth)
#     x, y = 0, 0
#     heading = angle
#     path_points = [(x, y)]
#     arc_types = ["F"]

#     def add_arc(x, y, heading, radius, extent_deg, steps=20):
#         """Compute points along a circular arc (like turtle.circle)"""
#         points = []
#         step_angle = radians(extent_deg / steps)
#         for i in range(steps):
#             theta1 = radians(heading) + i * step_angle
#             theta2 = radians(heading) + (i + 1) * step_angle
#             x_next = x + radius * (sin(theta2) - sin(theta1))
#             y_next = y - radius * (cos(theta2) - cos(theta1))
#             points.append((x_next, y_next))
#             x, y = x_next, y_next
#         heading += extent_deg
#         return points, heading, (x, y)

#     for ch in state:
#         if ch == 'F':
#             x2 = x + step * cos(radians(heading))
#             y2 = y + step * sin(radians(heading))
#             path_points.append((x2, y2))
#             arc_types.append("F")
#             x, y = x2, y2

#         elif ch == 'A':
#             # Keep arcs exactly as before
#             arc_points, heading, (x, y) = add_arc(x, y, heading, radius=step*1.2, extent_deg=90)
#             path_points.extend(arc_points)
#             arc_types.extend(["A"] * len(arc_points))

#         elif ch == 'B':
#             I = step / sqrt(2)
#             x2 = x + I * cos(radians(heading))
#             y2 = y + I * sin(radians(heading))
#             path_points.append((x2, y2))
#             arc_types.append("B")
#             x, y = x2, y2

#             arc_points, heading, (x, y) = add_arc(x, y, heading, radius=I*1.1, extent_deg=270)
#             path_points.extend(arc_points)
#             arc_types.extend(["B"] * len(arc_points))

#             x2 = x + I * cos(radians(heading))
#             y2 = y + I * sin(radians(heading))
#             path_points.append((x2, y2))
#             arc_types.append("B")
#             x, y = x2, y2

#         elif ch == 'L':
#             heading += 45
#         elif ch == 'R':
#             heading -= 45

#     return path_points, arc_types

# def draw_kolam_from_seed(seed="FBFBFBFB", depth=1, step=20, angle=0, img_size=(800, 800), padding=20):
#     path_points, arc_types = simulate_kolam_path(seed, depth, step, angle)

#     xs = [p[0] for p in path_points]
#     ys = [p[1] for p in path_points]
#     min_x, max_x = min(xs), max(xs)
#     min_y, max_y = min(ys), max(ys)

#     scale_x = (img_size[0] - 2*padding) / (max_x - min_x) if max_x != min_x else 1
#     scale_y = (img_size[1] - 2*padding) / (max_y - min_y) if max_y != min_y else 1
#     scale = min(scale_x, scale_y)

#     offset_x = padding - min_x * scale + (img_size[0] - 2*padding - (max_x - min_x) * scale)/2
#     offset_y = padding - min_y * scale + (img_size[1] - 2*padding - (max_y - min_y) * scale)/2

#     img = Image.new("RGB", img_size, "white")
#     draw = ImageDraw.Draw(img)

#     def transform(p):
#         return (p[0]*scale + offset_x, p[1]*scale + offset_y)

#     # Draw path
#     for i in range(1, len(path_points)):
#         t = arc_types[i-1]
#         color = 'darkgreen'
#         if t == "F":
#             width = max(2, int(scale*0.15))  # thin forward lines
#         elif t == "A":
#             width = max(3, int(scale*0.25))  # thicker arcs
#         elif t == "B":
#             width = max(3, int(scale*0.25))  # thicker arcs
#         draw.line((*transform(path_points[i-1]), *transform(path_points[i])), fill=color, width=width)

#     buf = io.BytesIO()
#     img.save(buf, format="PNG")
#     buf.seek(0)
#     return buf.getvalue()
