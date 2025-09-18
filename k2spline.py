import argparse
import pandas as pd
import numpy as np
from scipy.interpolate import splprep, splev

def csv_to_desmos(csv_path, num_samples=200):
    # Load CSV
    df = pd.read_csv(csv_path)
    x = df.iloc[:,0].dropna().to_numpy()
    y = df.iloc[:,1].dropna().to_numpy()

    # Normalize to [-10,10] range for Desmos
    x = (x - np.mean(x)) / np.ptp(x) * 10
    y = (y - np.mean(y)) / np.ptp(y) * 10

    # Parametric spline (cubic B-spline)
    tck, u = splprep([x, y], s=0)
    u_new = np.linspace(0, 1, num_samples)
    x_new, y_new = splev(u_new, tck)

    # Convert samples into Desmos-style parametric list
    desmos_points = ", ".join([f"({x_new[i]:.3f},{y_new[i]:.3f})" for i in range(len(x_new))])
    desmos_expr = f"{{ {desmos_points} }}"

    print("✅ Copy this into Desmos (as a table or parametric curve):")
    print(desmos_expr)

    return desmos_expr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    csv_to_desmos(args.csv)