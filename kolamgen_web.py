import io
from math import cos, sin, radians, sqrt, pi
from PIL import Image, ImageDraw
from lsystem import generate_lsystem_state

class WebTurtle:
    """A turtle-like class that tracks position and drawing operations for PIL Image rendering"""
    
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0  # in degrees
        self.pen_down = True
        self.pen_size = 3
        self.pen_color = 'black'
        self.path_segments = []  # List of (start_pos, end_pos, color, width) tuples
        self.dot_positions = []  # List of (x, y, size, color) tuples
        self.enclosed_regions = []  # Track enclosed shapes for center dots
        self.current_shape_points = []  # Points in current shape being drawn
    
    def penup(self):
        self.pen_down = False
    
    def pendown(self):
        self.pen_down = True
    
    def pensize(self, size):
        self.pen_size = size
    
    def color(self, color):
        self.pen_color = color
    
    def setheading(self, angle):
        self.heading = angle
    
    def position(self):
        return (self.x, self.y)
    
    def goto(self, x, y):
        if self.pen_down:
            self.path_segments.append(((self.x, self.y), (x, y), self.pen_color, self.pen_size))
        self.x = x
        self.y = y
    
    def forward(self, distance):
        """Move forward by distance units"""
        new_x = self.x + distance * cos(radians(self.heading))
        new_y = self.y + distance * sin(radians(self.heading))
        if self.pen_down:
            self.path_segments.append(((self.x, self.y), (new_x, new_y), self.pen_color, self.pen_size))
            # Track points for enclosed region detection
            self.current_shape_points.append((self.x, self.y))
            self.current_shape_points.append((new_x, new_y))
        self.x = new_x
        self.y = new_y
    
    def fd(self, distance):
        """Alias for forward"""
        self.forward(distance)
    
    def left(self, angle):
        """Turn left by angle degrees"""
        self.heading += angle
    
    def right(self, angle):
        """Turn right by angle degrees"""
        self.heading -= angle
    
    def circle(self, radius, extent):
        """Draw a circular arc. Positive radius = counterclockwise, negative = clockwise"""
        if not self.pen_down:
            # Just move without drawing
            self._move_along_arc(radius, extent)
            return
        
        # Track start position for enclosed region detection
        arc_start_x, arc_start_y = self.x, self.y
        arc_points = [(arc_start_x, arc_start_y)]
        
        # Calculate the arc in small segments for smooth drawing
        num_segments = max(8, int(abs(extent) / 10))  # At least 8 segments, more for larger arcs
        angle_step = extent / num_segments
        
        for i in range(num_segments):
            start_x, start_y = self.x, self.y
            self._move_along_arc(radius, angle_step)
            self.path_segments.append(((start_x, start_y), (self.x, self.y), self.pen_color, self.pen_size))
            arc_points.append((self.x, self.y))
        
        # Add all arc points to current shape
        self.current_shape_points.extend(arc_points)
        
        # Check if this completes a significant enclosed region (like a full circle or large arc)
        if abs(extent) >= 270:  # Nearly complete circle or loop
            center = self._calculate_arc_center(arc_start_x, arc_start_y, radius, extent)
            if center:
                self.enclosed_regions.append(center)
    
    def _move_along_arc(self, radius, extent):
        """Move the turtle along an arc without drawing"""
        # Convert to radians
        extent_rad = radians(extent)
        heading_rad = radians(self.heading)
        
        # Calculate the center of the circle
        if radius > 0:
            # Counterclockwise circle, center is to the left
            center_x = self.x - radius * sin(heading_rad)
            center_y = self.y + radius * cos(heading_rad)
        else:
            # Clockwise circle, center is to the right
            radius = abs(radius)
            center_x = self.x + radius * sin(heading_rad)
            center_y = self.y - radius * cos(heading_rad)
        
        # Calculate new position after the arc
        if extent >= 0:
            # Counterclockwise
            new_heading_rad = heading_rad + extent_rad
            self.x = center_x + radius * sin(new_heading_rad)
            self.y = center_y - radius * cos(new_heading_rad)
            self.heading += extent
        else:
            # Clockwise
            new_heading_rad = heading_rad + extent_rad
            self.x = center_x - radius * sin(new_heading_rad)
            self.y = center_y + radius * cos(new_heading_rad)
            self.heading += extent
    
    def dot(self, size, color):
        """Place a dot at the current position"""
        self.dot_positions.append((self.x, self.y, size, color))
    
    def _calculate_arc_center(self, start_x, start_y, radius, extent):
        """Calculate the center of an arc for enclosed region detection"""
        try:
            heading_rad = radians(self.heading - extent)  # Heading at start of arc
            
            if radius > 0:
                # Counterclockwise arc, center is to the left of start direction
                center_x = start_x - abs(radius) * sin(heading_rad)
                center_y = start_y + abs(radius) * cos(heading_rad)
            else:
                # Clockwise arc, center is to the right of start direction
                center_x = start_x + abs(radius) * sin(heading_rad)
                center_y = start_y - abs(radius) * cos(heading_rad)
            
            return (center_x, center_y)
        except:
            return None
    
    def start_new_shape(self):
        """Mark the start of a new potential enclosed shape"""
        self.current_shape_points = []
    
    def check_for_enclosed_region(self):
        """Check if current shape points form an enclosed region and add center dot"""
        if len(self.current_shape_points) < 6:  # Need at least a triangle
            return
        
        # Check if the shape is roughly closed (start and end points are close)
        if len(self.current_shape_points) >= 2:
            start_point = self.current_shape_points[0]
            end_point = self.current_shape_points[-1]
            distance = sqrt((start_point[0] - end_point[0])**2 + (start_point[1] - end_point[1])**2)
            
            # If the shape is closed (within tolerance), calculate centroid
            if distance < 20:  # Tolerance for considering it "closed"
                center = self._calculate_centroid(self.current_shape_points)
                if center:
                    self.enclosed_regions.append(center)
    
    def _calculate_centroid(self, points):
        """Calculate the centroid of a set of points"""
        if not points:
            return None
        
        # Remove duplicate points
        unique_points = []
        for point in points:
            if not unique_points or (abs(point[0] - unique_points[-1][0]) > 1 or 
                                   abs(point[1] - unique_points[-1][1]) > 1):
                unique_points.append(point)
        
        if len(unique_points) < 3:
            return None
        
        # Simple centroid calculation
        center_x = sum(p[0] for p in unique_points) / len(unique_points)
        center_y = sum(p[1] for p in unique_points) / len(unique_points)
        
        return (center_x, center_y)


def draw_kolam_web(seed="FBFBFBFB", depth=1, step=20, angle=0, img_size=(800, 800), padding=50):
    # Generate L-system state
    if isinstance(seed, str) and all(c in "FABLRC" for c in seed):
        state = generate_lsystem_state(seed, depth)
    else:
        # If seed is already an L-system state string, use it directly
        state = seed
    
    # Create web turtle
    t = WebTurtle()
    t.pendown()
    t.setheading(angle)
    
    # Track dot positions and colorful mode
    dot_positions = []
    is_colorful = False  # Start in white mode, only 'C' command can enable colors
    
    for ch in state:
        if ch == 'F':
            t.color("#07DA07" if is_colorful else 'white')
            t.pensize(3)
            t.start_new_shape()  # Start tracking for potential enclosed region
            t.fd(step)
            
        elif ch == 'A':
            t.color("#009DFF" if is_colorful else 'white')
            t.pensize(3)
            
            dot_positions.append(t.position())
            
            t.start_new_shape()  # Start tracking the arc as a potential enclosed region
            t.circle(step * 1.5, 90)  # Quarter-circle arc (exact same parameters)
            
        elif ch == 'B':
            t.color('red' if is_colorful else 'white')
            t.pensize(3)
            
            dot_positions.append(t.position())

            I = step / sqrt(2)
            t.start_new_shape()  # Start tracking the B-shape
            t.fd(I)
            t.circle(I * 0.7, 270)  # This will automatically detect the enclosed loop
            t.fd(I)
            t.check_for_enclosed_region()  # Check if we completed an enclosed region
            
        elif ch == 'C':
            is_colorful = not is_colorful  # Toggle colorful mode
            
        elif ch == 'L':
            t.left(45)  # Rotate left 45° without drawing
            t.check_for_enclosed_region()  # Check for completed shapes at turns
            
        elif ch == 'R':
            t.right(45)  # Rotate right 45° without drawing  
            t.check_for_enclosed_region()  # Check for completed shapes at turns

    
    # Draw dots at collected positions (same as turtle version)
    unique_dot_positions = set(dot_positions)
    t.penup()
    
    for x, y in unique_dot_positions:
        t.goto(x, y)
        t.dot(6, 'black' if is_colorful else 'white')
    
    # Add center dots for enclosed regions
    for center in t.enclosed_regions:
        t.dot_positions.append((center[0], center[1], 2, 'darkred' if is_colorful else 'white'))  # Smaller dots for centers
    
    # Now render everything to a PIL image
    return render_to_image(t, img_size, padding)


def render_to_image(turtle, img_size=(800, 800), padding=50):
    """Convert turtle path data to a PIL Image"""
    
    # Get all points for bounds calculation
    all_points = []
    for segment in turtle.path_segments:
        all_points.extend([segment[0], segment[1]])
    for dot in turtle.dot_positions:
        all_points.append((dot[0], dot[1]))
    
    if not all_points:
        # Empty image if no drawing occurred
        img = Image.new("RGB", img_size, "black")
        return img
    
    # Calculate bounds
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Calculate scale and offset to fit in image with padding
    width = max_x - min_x
    height = max_y - min_y
    
    if width == 0 and height == 0:
        scale = 1
    else:
        scale_x = (img_size[0] - 2*padding) / width if width > 0 else 1
        scale_y = (img_size[1] - 2*padding) / height if height > 0 else 1
        scale = min(scale_x, scale_y)
    
    # Calculate offsets to center the drawing
    offset_x = padding - min_x * scale + (img_size[0] - 2*padding - width * scale) / 2
    offset_y = padding - min_y * scale + (img_size[1] - 2*padding - height * scale) / 2
    
    def transform_point(x, y):
        """Transform from turtle coordinates to image coordinates"""
        return (x * scale + offset_x, y * scale + offset_y)
    
    # Create image
    img = Image.new("RGB", img_size, "black")
    draw = ImageDraw.Draw(img)
    
    # Draw all path segments
    for segment in turtle.path_segments:
        start_pos, end_pos, color, width = segment
        start_x, start_y = transform_point(*start_pos)
        end_x, end_y = transform_point(*end_pos)
        
        # Scale line width
        scaled_width = max(1, int(width * scale * 0.6))
        
        draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=scaled_width)
    
    # Draw all dots
    for dot in turtle.dot_positions:
        x, y, size, color = dot
        tx, ty = transform_point(x, y)
        scaled_size = max(2, int(size * scale * 0.2))
        
        # Draw filled circle
        draw.ellipse([tx - scaled_size//2, ty - scaled_size//2, 
                     tx + scaled_size//2, ty + scaled_size//2], 
                    fill=color)
    
    return img


def draw_kolam_web_bytes(seed="FBFBFBFB", depth=1, step=20, angle=0, img_size=(800, 800), padding=50):
    """
    Generate kolam and return as PNG bytes (for web API)
    Color mode is controlled by 'C' commands within the seed string.
    """
    img = draw_kolam_web(seed, depth, step, angle, img_size, padding)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def generate_drawing_steps(seed="FBFBFBFB", depth=1, step=20, angle=0, canvas_size=(800, 600)):
    """
    Generate step-by-step drawing instructions for canvas animation
    Returns a list of drawing commands that can be executed sequentially
    """
    # Generate L-system state
    if isinstance(seed, str) and all(c in "FABLRC" for c in seed):
        state = generate_lsystem_state(seed, depth)
    else:
        state = seed
    
    # Create web turtle to trace the path
    t = WebTurtle()
    t.pendown()
    t.setheading(angle)
    
    # Track drawing steps and dots
    drawing_steps = []
    dot_positions = []
    is_colorful = False  # Start in white mode
    
    # Execute L-system commands and record steps
    for ch in state:
        if ch == 'F':
            color = '#00FF00' if is_colorful else 'white'
            start_pos = (t.x, t.y)
            t.fd(step)
            end_pos = (t.x, t.y)
            
            drawing_steps.append({
                'type': 'line',
                'start': {'x': start_pos[0], 'y': start_pos[1]},
                'end': {'x': end_pos[0], 'y': end_pos[1]},
                'color': color,
                'width': 3
            })
            
        elif ch == 'A':
            color = "#1B95EC" if is_colorful else 'white'
            
            # Store position for dot (same as main draw function)
            dot_positions.append(t.position())
            
            # Store position for animation
            start_pos = t.position()
            start_heading = t.heading
            
            # First, call the actual turtle method to detect enclosed regions
            t.start_new_shape()  # Start tracking the arc as a potential enclosed region
            t.circle(step * 1.5, 90)
            
            # Now reset position and create animation segments
            t.x, t.y = start_pos
            t.heading = start_heading
            
            # Simulate the arc with small steps for animation
            arc_steps = 20
            radius = step * 1.5
            for i in range(arc_steps):
                current_pos = (t.x, t.y)
                t._move_along_arc(radius, 90 / arc_steps)
                new_pos = (t.x, t.y)
                
                drawing_steps.append({
                    'type': 'line',
                    'start': {'x': current_pos[0], 'y': current_pos[1]},
                    'end': {'x': new_pos[0], 'y': new_pos[1]},
                    'color': color,
                    'width': 3
                })
            
        elif ch == 'B':
            color = 'red' if is_colorful else 'white'
            
            # Store position for dot (same as main draw function)
            dot_positions.append(t.position())
            
            I = step / sqrt(2)
            
            # Store starting position for both region detection and animation
            region_start_pos = t.position()
            region_start_heading = t.heading
            
            # First, call the actual turtle methods to detect enclosed regions
            t.start_new_shape()
            t.fd(I)
            t.circle(I * 0.7, 270)
            t.fd(I)
            t.check_for_enclosed_region()
            
            # Now reset position and create animation segments
            t.x, t.y = region_start_pos
            t.heading = region_start_heading
            
            # First forward movement for animation
            start_pos = (t.x, t.y)
            t.fd(I)
            end_pos = (t.x, t.y)
            drawing_steps.append({
                'type': 'line',
                'start': {'x': start_pos[0], 'y': start_pos[1]},
                'end': {'x': end_pos[0], 'y': end_pos[1]},
                'color': color,
                'width': 3
            })
            
            # Arc with multiple segments for animation
            arc_steps = 30
            radius = I * 0.7
            for i in range(arc_steps):
                current_pos = (t.x, t.y)
                t._move_along_arc(radius, 270 / arc_steps)
                new_pos = (t.x, t.y)
                
                drawing_steps.append({
                    'type': 'line',
                    'start': {'x': current_pos[0], 'y': current_pos[1]},
                    'end': {'x': new_pos[0], 'y': new_pos[1]},
                    'color': color,
                    'width': 3
                })
            
            # Second forward movement for animation
            start_pos = (t.x, t.y)
            t.fd(I)
            end_pos = (t.x, t.y)
            drawing_steps.append({
                'type': 'line',
                'start': {'x': start_pos[0], 'y': start_pos[1]},
                'end': {'x': end_pos[0], 'y': end_pos[1]},
                'color': color,
                'width': 3
            })
            
        elif ch == 'C':
            is_colorful = not is_colorful  # Toggle colorful mode
            
        elif ch == 'L':
            t.left(45)
            t.check_for_enclosed_region()  # Check for completed shapes at turns
            
        elif ch == 'R':
            t.right(45)
            t.check_for_enclosed_region()  # Check for completed shapes at turns
    
    # Calculate scaling and positioning for canvas
    scale = 1
    offset_x = 0
    offset_y = 0
    
    if drawing_steps:
        all_x = []
        all_y = []
        for step in drawing_steps:
            if step['type'] == 'line':
                all_x.extend([step['start']['x'], step['end']['x']])
                all_y.extend([step['start']['y'], step['end']['y']])
        
        if all_x and all_y:
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            
            # Calculate scale and offset
            width = max_x - min_x
            height = max_y - min_y
            padding = 50
            
            if width > 0 and height > 0:
                scale_x = (canvas_size[0] - 2*padding) / width
                scale_y = (canvas_size[1] - 2*padding) / height
                scale = min(scale_x, scale_y)
                
                offset_x = padding - min_x * scale + (canvas_size[0] - 2*padding - width * scale) / 2
                offset_y = padding - min_y * scale + (canvas_size[1] - 2*padding - height * scale) / 2
                
                # Transform all coordinates
                for step in drawing_steps:
                    if step['type'] == 'line':
                        step['start']['x'] = step['start']['x'] * scale + offset_x
                        step['start']['y'] = step['start']['y'] * scale + offset_y
                        step['end']['x'] = step['end']['x'] * scale + offset_x
                        step['end']['y'] = step['end']['y'] * scale + offset_y
                        step['width'] = max(1, int(step['width'] * scale * 0.6))
    
    # Add dots at the end
    # Process basic dot positions (from A and B commands)
    unique_dot_positions = set(dot_positions)
    for x, y in unique_dot_positions:
        transformed_x = x * scale + offset_x
        transformed_y = y * scale + offset_y
        dot_color = 'white'  # Always white for visibility on black canvas
        
        drawing_steps.append({
            'type': 'dot',
            'x': transformed_x,
            'y': transformed_y,
            'radius': 1,  # Smaller dots
            'color': dot_color
        })
    
    # Add center dots for enclosed regions (this was missing!)
    for center in t.enclosed_regions:
        t.dot_positions.append((center[0], center[1], 2, 'white'))  # Always white for dark theme, smaller size
    
    # Process turtle's dot positions (from dot() method calls with color info)
    for x, y, size, color in t.dot_positions:
        transformed_x = x * scale + offset_x
        transformed_y = y * scale + offset_y
        # Ensure dot color is visible on black canvas
        if color == 'black' or color == 'darkred':
            dot_color = 'white'
        else:
            dot_color = color
        
        drawing_steps.append({
            'type': 'dot',
            'x': transformed_x,
            'y': transformed_y,
            'radius': max(1, int(size * scale * 0.2)),  # Smaller dots with reduced scaling
            'color': dot_color
        })
    
    return drawing_steps