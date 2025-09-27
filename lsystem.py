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
Rule 7: Kolam is completed when all points are enclosed by the drawing line
Rule 8: Color control commands affect rendering but do not alter spatial geometry

- C → C (Color toggle, passes through unchanged)

Color Command (C) Placement Rules:
1. Position Independence: C can be placed anywhere in the L-system string
2. Non-Spatial: C does not affect turtle position, heading, or geometric structure
3. State Modifier: C toggles between white mode (default) and colorful mode
4. Persistence: Color state persists until next C command is encountered
5. Expansion Immunity: C remains unchanged during L-system iterations
6. Multiple Usage: Multiple C commands create alternating color segments
7. Strategic Placement:
   - Beginning (CFBFB): Entire pattern starts colorful
   - Middle (FBCFB): Color transition partway through
   - Symmetric (CFBCFBC): Balanced color alternation
   - End (FBFBC): Ends with color change (affects recursive expansions)

Example Color Patterns:
- "FBFBFBFB" → All white strokes (default)
- "CFBFBFBFB" → All colorful strokes
- "FBCFBCFB" → White-Color-White alternation
- "CABACABA" → Colorful arcs and petals only
- "FCBCACBC" → Color toggle before each different element type

Advanced Placement Strategies:
1. Sectional Coloring: Place C before major structural elements (A, B)
2. Rhythmic Patterns: Regular C intervals for breathing color rhythm
3. Hierarchical Coloring: Different colors for different recursion levels
4. Symmetrical Balance: Mirror C placement for visual harmony
5. Emphasis Control: Use C to highlight specific pattern features 
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
            elif ch == 'C':
                final_state += 'C'  # C command passes through unchanged
            else:
                final_state += ch  # Pass through other commands (L, R, etc.)
        state = final_state
    return state

__all__ = ['generate_lsystem_state']