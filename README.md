# KolamCreate

SIH 25107 - Develop computer programs (in any language, preferably Python) to identify the design principles behind the Kolam designs and recreate the kolams.

Kolams (known by other narnes as muggu, rangoli and rangavalli) are significant cultural traditions of India, blending art, ingenuity, and culture.

The designs vary by region, and the designs consist of grids of dots, with symmetry, repetition, and spatial reasoning embedded in them.

The Kolam designs provide a fascinating area of study for their strong mathematical underpinnings.

The challenge is to develop computer programs (in any language, preferably Python) to identify the design principles behind the Kolam designs and recreate the kolams.


## Manim Installation

1. brew install pkg-config cairo pango gdk-pixbuf libffi
2. pip install manim


## Kolam Rules
Rule 1: Uniformly spacing of dots
Rule 2: Smooth drawing line around the dots
Rule 3: Symmetry in drawings
Rule 4: Straight lines are drawn inclined at an angle of 45 degrees
Rule 5: Drawing lines never trace back
Rule 6: Arcs adjoining the dots
Rule 7 : Kolam is completed when all points are enclosed by the drawing line

## Execution
1. (First-Time) python3 -m venv venv
2. (First-Time) pip install -r requirements.txt
3. To Run the Turtle Version: python kolam_turtle.py
4. To Run the Manim Version: manim -pql kolam_manim.py Kolam