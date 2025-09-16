# kolam_recreate.py (updated)
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev

def draw_from_extraction(pkl_file, out_prefix="kolam_recreated",
                         dpi=300, scale=1.0, show=True):
    """Load a kolam extraction pickle and recreate the design with auto-scaling."""
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    dots = data.get("dots", np.zeros((0, 2)))
    fits = data.get("fits", [])

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("black")   # background for entire figure
    ax.set_facecolor("black")          # background for axes


    # ---- collect all coordinates for automatic scaling ----
    all_x, all_y = [], []

    # draw curves
    for (_uv, fit) in fits:
        if fit.get("type") == "spline" and "tck" in fit:
            u = np.linspace(0, 1, 400)
            x, y = splev(u, fit["tck"])
            all_x.extend(x)
            all_y.extend(y)
            ax.plot(x * scale, y * scale, color="white", lw=1.5)
        elif "pts" in fit:
            pts = np.array(fit["pts"])
            if pts.size:
                all_x.extend(pts[:, 0])
                all_y.extend(pts[:, 1])
                ax.plot(pts[:, 0] * scale, pts[:, 1] * scale, color="white", lw=1.5)

    # draw dots
    if dots.size:
        all_x.extend(dots[:, 0])
        all_y.extend(dots[:, 1])
        ax.scatter(dots[:, 0] * scale, dots[:, 1] * scale,
                   s=35, facecolors="yellow",
                   edgecolors="black", linewidths=0.6, zorder=5)

    # ---- auto-scale axes ----
    if all_x and all_y:
        margin = max(np.ptp(all_x), np.ptp(all_y)) * 0.05  # 5% margin
        ax.set_xlim(min(all_x)*scale - margin, max(all_x)*scale + margin)
        ax.set_ylim(min(all_y)*scale - margin, max(all_y)*scale + margin)

    # ---- save images ----
    out_png = f"{out_prefix}.png"
    out_svg = f"{out_prefix}.svg"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0)
    fig.savefig(out_svg, dpi=dpi, bbox_inches="tight", pad_inches=0)
    print(f"Saved: {out_png}, {out_svg}")

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Recreate kolam from extraction pickle")
    ap.add_argument("pkl", help="Input extraction pickle file")
    ap.add_argument("--out", "-o", default="kolam_recreated", help="Output file prefix")
    ap.add_argument("--dpi", type=int, default=300, help="DPI for saved image")
    ap.add_argument("--scale", type=float, default=1.0, help="Scale factor for geometry")
    ap.add_argument("--no-show", action="store_true", help="Do not display plot window")
    args = ap.parse_args()

    draw_from_extraction(
        args.pkl,
        out_prefix=args.out,
        dpi=args.dpi,
        scale=args.scale,
        show=not args.no_show,
    )