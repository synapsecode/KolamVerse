"""
Lindermayer system is a parallel rewriting system and a type of formal grammar. 
It consists of an alphabet of symbols that can be used to make strings, a collection of production rules that expand 
each symbol into some larger string of symbols.
The recursive nature of L system rules leads to self similarity and thereby fractal like forms are easy to describe 
with an L system. This nature is applied in generating kolam pattern. Kolam pattern becomes more complex by increasing 
the iteration level.

Kolam Rules:
Rule 1: Uniformly spacing of dots
Rule 2: Smooth drawing line around the dots
Rule 3: Symmetry in drawings
Rule 4: Straight lines are drawn inclined at an angle of 45 degrees
Rule 5: Drawing lines never trace back
Rule 6: Arcs adjoining the dots
Rule 7 : Kolam is completed when all points are enclosed by the drawing line
"""

def generate_lsystem_state(state, n):
    for _ in range(n):
        final_state = ''
        for ch in state:
            if ch == 'F':
                final_state += 'F'
            elif ch == 'A':
                final_state += 'AFBFA'
            elif ch == 'B':
                final_state += 'AFBFBFBFA'
        state = final_state
    return state

__all__ = ['generate_lsystem_state']