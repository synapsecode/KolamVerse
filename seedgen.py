import argparse
import itertools
import math

"""
Turn sums = ±360°, orientation returns to original.
Movement also returns to original position.
"""

# Direction mapping: heading in degrees → (dx, dy)
DIRECTIONS = {
    0: (1, 0),    # East
    90: (0, 1),   # North
    180: (-1, 0), # West
    270: (0, -1), # South
}

def get_turn(symbol):
    """Return the turn in degrees for each symbol."""
    if symbol == 'A':
        return 90
    elif symbol == 'B':
        return -90
    else:  # 'F'
        return 0

def simulate(seed):
    """Simulate the turtle's movement and check if it returns to start."""
    x, y = 0, 0
    heading = 0  # start pointing east

    for s in seed:
        turn = get_turn(s)
        heading = (heading + turn) % 360
        
        if s == 'F':
            dx, dy = DIRECTIONS[heading]
            x += dx
            y += dy
        elif s == 'A':
            # For arcs, assume turtle moves along a quarter circle, ending at rotated direction
            dx, dy = DIRECTIONS[heading]
            x += dx
            y += dy
        elif s == 'B':
            dx, dy = DIRECTIONS[heading]
            x += dx
            y += dy

    # Check if turtle returned to start position and heading
    return x == 0 and y == 0 and heading == 0

def generate_valid_seeds(length):
    """Generate all valid seeds for a given even length."""
    if length % 2 != 0:
        raise ValueError("Only even lengths supported.")
    
    half_len = length // 2
    valid_seeds = []
    symbols = ['A', 'B']

    # Generate all combinations for half the sequence
    for half in itertools.product(symbols, repeat=half_len):
        # Mirror to form full seed
        full_seed = ''.join(half + half[::-1])
        
        if simulate(full_seed):
            valid_seeds.append(full_seed)
    return valid_seeds

def main(length):
    seeds = generate_valid_seeds(length)
    
    if seeds:
        print(f"Found {len(seeds)} valid seeds:")
        for s in seeds:
            print(s)
    else:
        print("No valid seeds found.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--length', default=8)
    args = ap.parse_args()
    main(int(args.length))
