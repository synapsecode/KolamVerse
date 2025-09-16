from manim import *
from math import sqrt, cos, sin, radians
import numpy as np
from lsystem import generate_lsystem_state

class KolamScene(Scene):
    def draw_kolam(self, state, step=0.5):  # Adjusted step size for Manim's scale
        dot_positions = []
        current_pos = np.array([0, 0, 0])  # Starting at origin (x, y, 0)
        current_angle = 0  # In degrees, initially facing right
        elements = VGroup()  # To store all lines, arcs, and dots

        # Conversion factor for Turtle's step to Manim's coordinate system
        scale = step

        for ch in state:
            if ch == 'F':
                # Calculate end position for straight line
                end_pos = current_pos + np.array([
                    scale * cos(radians(current_angle)),
                    scale * sin(radians(current_angle)),
                    0
                ])
                # Draw green line
                line = Line(start=current_pos, end=end_pos, color=GREEN, stroke_width=3)
                elements.add(line)
                current_pos = end_pos

            elif ch == 'A':
                # Store position for dot
                dot_positions.append(current_pos.copy())
                # Draw blue arc (90-degree arc with radius 1.5 * step)
                radius = scale * 1.5
                arc = Arc(
                    radius=radius,
                    start_angle=radians(current_angle),
                    angle=radians(90),
                    color=BLUE,
                    stroke_width=3
                )
                # Position the arc correctly
                arc_center = current_pos + np.array([
                    radius * cos(radians(current_angle + 90)),
                    radius * sin(radians(current_angle + 90)),
                    0
                ])
                arc.shift(arc_center - arc.get_center())
                elements.add(arc)
                # Update position and angle
                current_angle += 90
                current_pos = arc_center + np.array([
                    radius * cos(radians(current_angle)),
                    radius * sin(radians(current_angle)),
                    0
                ])

            elif ch == 'B':
                # Store position for dot
                dot_positions.append(current_pos.copy())
                # B pattern: forward, 270-degree arc, forward
                I = scale / sqrt(2)
                # First forward segment
                end_pos = current_pos + np.array([
                    I * cos(radians(current_angle)),
                    I * sin(radians(current_angle)),
                    0
                ])
                line1 = Line(start=current_pos, end=end_pos, color=RED, stroke_width=3)
                elements.add(line1)
                current_pos = end_pos
                # 270-degree arc
                radius = I * 0.7
                arc = Arc(
                    radius=radius,
                    start_angle=radians(current_angle),
                    angle=radians(270),
                    color=RED,
                    stroke_width=3
                )
                arc_center = current_pos + np.array([
                    radius * cos(radians(current_angle + 90)),
                    radius * sin(radians(current_angle + 90)),
                    0
                ])
                arc.shift(arc_center - arc.get_center())
                elements.add(arc)
                current_angle += 270
                current_pos = arc_center + np.array([
                    radius * cos(radians(current_angle)),
                    radius * sin(radians(current_angle)),
                    0
                ])
                # Second forward segment
                end_pos = current_pos + np.array([
                    I * cos(radians(current_angle)),
                    I * sin(radians(current_angle)),
                    0
                ])
                line2 = Line(start=current_pos, end=end_pos, color=RED, stroke_width=3)
                elements.add(line2)
                current_pos = end_pos

        # Add unique dots
        unique_dot_positions = set(tuple(pos[:2]) for pos in dot_positions)  # Use 2D tuples for uniqueness
        for x, y in unique_dot_positions:
            dot = Dot(point=np.array([x, y, 0]), radius=0.05, color=BLACK)
            elements.add(dot)

        return elements

    def construct(self):
        # Hardcode L-system level (can be modified or made configurable)
        n = 3  # Recommended: 2-5 for reasonable complexity
        initial_state = 'FBFBFBFB'
        state = generate_lsystem_state(initial_state, n)

        # Draw the Kolam
        kolam = self.draw_kolam(state, step=0.5)
        self.add(kolam)

        # Center the camera on the Kolam
        # self.camera.frame.move_to(kolam.get_center())
        self.wait(2)  # Display for 2 seconds