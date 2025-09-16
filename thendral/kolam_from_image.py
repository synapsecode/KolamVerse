# kolam_from_image.py
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy import ndimage
from scipy.interpolate import splprep, splev
import networkx as nx
from sklearn.neighbors import KDTree
from shapely.geometry import Point
from math import cos, sin, radians, atan2, degrees

# -------------------------
# Utility / preprocessing
# -------------------------
def read_and_preprocess(path, show=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    # normalize and invert so strokes = 1
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary = (th == 0).astype(np.uint8)  # strokes black -> 1
    # slight blur + morphological closing to join broken strokes
    binary = cv2.medianBlur((binary*255).astype(np.uint8), 3)//255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if show:
        plt.figure(); plt.title("Binary (strokes=1)"); plt.imshow(binary, cmap='gray'); plt.axis('off')
    return binary

# -------------------------
# Dot (pulli) detection
# -------------------------
def detect_dots(original_gray, binary_strokes):
    # Dots usually are separate small blobs; we find connected components on inverted strokes
    # create image with dots highlighted by distance transform local maxima
    inv = 1 - binary_strokes
    dist = ndimage.distance_transform_edt(inv)
    # find peaks
    coords = np.column_stack(ndimage.maximum_position(dist))
    # fallback: connected components of inv
    labels, n = ndimage.label(inv)
    centers = []
    for lab in range(1, n+1):
        ys, xs = np.where(labels==lab)
        if len(xs)==0: continue
        cx = xs.mean(); cy = ys.mean()
        centers.append((cx, cy))
    # dedupe and return
    pts = np.array(centers)
    if pts.size==0:
        return np.zeros((0,2))
    # optionally cluster to remove near duplicates
    tree = KDTree(pts)
    keep = []
    used = set()
    for i,p in enumerate(pts):
        if i in used: continue
        idx = tree.query_radius([p], r=3)[0]
        used.update(idx.tolist())
        keep.append(pts[idx].mean(axis=0))
    return np.array(keep)

# -------------------------
# Skeletonize strokes and extract paths / junctions
# -------------------------# -------------------------
# Skeletonize strokes and extract paths / junctions
# -------------------------
def skeleton_and_graph(binary_strokes, show=False):
    # skeletonize expects boolean image
    skel = skeletonize(binary_strokes==1).astype(np.uint8)
    if show:
        plt.figure(); plt.title("skeleton"); plt.imshow(skel, cmap='gray'); plt.axis('off')

    H, W = skel.shape

    # neighbors in (x,y) ordering -> (col, row)
    def neighbors_xy(x, y):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                    yield (nx, ny)   # return (x, y) style

    # classify pixels: endpoints (degree 1) and junctions (degree > 2)
    endpoints = []
    junctions = []
    for y in range(H):
        for x in range(W):
            if not skel[y, x]:
                continue
            deg = sum(1 for _ in neighbors_xy(x, y))
            if deg == 1:
                endpoints.append((x, y))
            elif deg > 2:
                junctions.append((x, y))

    # anchor points are endpoints + junctions
    node_points = endpoints + junctions
    anchor_map = {pt: i for i, pt in enumerate(node_points)}

    # graph to store edges between anchors
    G = nx.Graph()
    for i, pt in enumerate(node_points):
        G.add_node(i, coord=(pt[0], pt[1]))

    # follow paths starting from an anchor to reach another anchor
    def follow_path(start):
        # start is (x,y) anchor
        x0, y0 = start
        path = [(x0, y0)]
        visited = set()
        visited.add((x0, y0))

        # explore each neighbor branch separately and return list of paths found from this anchor
        found_paths = []

        for nbr in neighbors_xy(x0, y0):
            # start a new branch walk
            branch = [(x0, y0)]
            prev = (x0, y0)
            cur = nbr
            branch_visited = set(branch)
            branch_visited.add(cur)
            branch.append(cur)

            # walk until we hit another anchor or dead end; limit steps to avoid infinite loop
            steps = 0
            MAX_STEPS = H * W  # safety cap
            while steps < MAX_STEPS:
                steps += 1
                cx, cy = cur
                # if cur is an anchor (and not the starting anchor), finish this branch
                if cur in anchor_map and cur != start:
                    found_paths.append(branch.copy())
                    break

                # get neighbors of current pixel excluding the previous pixel
                nbrs = [p for p in neighbors_xy(cx, cy) if p != prev]

                # remove already visited on this branch to avoid loops
                nbrs = [p for p in nbrs if p not in branch_visited]

                if not nbrs:
                    # dead end
                    break

                # choose first neighbor to continue (skeleton is thin; this suffices)
                nxt = nbrs[0]
                branch.append(nxt)
                branch_visited.add(nxt)
                prev = cur
                cur = nxt

            # end branch
        return found_paths

    # iterate over anchors and add edges for each path found
    for start in node_points:
        paths_from_anchor = follow_path(start)
        a = anchor_map[start]
        for branch in paths_from_anchor:
            end_pt = branch[-1]
            if end_pt not in anchor_map:
                continue
            b = anchor_map[end_pt]
            # build coords in (x,y) order for storage
            coords = [(int(p[0]), int(p[1])) for p in branch]
            # avoid duplicate edges (undirected)
            if not G.has_edge(a, b):
                G.add_edge(a, b, pixels=coords)

    return skel, G

# -------------------------
# Fit parametric spline to a path
# -------------------------
def fit_spline(path_pixels, smooth=1e-1, k=3):
    # path_pixels: list of (x,y)
    pts = np.array(path_pixels)
    if pts.shape[0] < 4:
        # fallback: poly fit
        x = pts[:,0]; y = pts[:,1]
        # attempt line fit
        A = np.vstack([x, np.ones_like(x)]).T
        m,c = np.linalg.lstsq(A, y, rcond=None)[0]
        return {'type':'line','m':m,'c':c,'pts':pts}
    # parametrize by arc length t
    d = np.sqrt(((np.diff(pts, axis=0))**2).sum(axis=1))
    t = np.hstack(([0], np.cumsum(d)))
    t = t / t[-1]
    try:
        tck, u = splprep([pts[:,0], pts[:,1]], u=t, s=smooth, k=min(k, pts.shape[0]-1))
        return {'type':'spline','tck':tck, 'pts':pts}
    except Exception as e:
        # fallback to piecewise linear
        return {'type':'polyline','pts':pts}

def spline_to_equation(tck, name="S"):
    # we cannot print closed-form; present parametric form with control points
    ctrlx, ctrly = tck[1]
    knots = tck[0]
    degree = tck[2]
    # return a readable representation
    return f"{name}(t): parametric B-spline of degree {degree} with {len(ctrlx)} control points. Use scipy.interpolate.splev to evaluate."

# -------------------------
# Fit circle (optional)
# -------------------------
def fit_circle(pts):
    # algebraic circle fit (Taubin)
    x = pts[:,0]; y = pts[:,1]
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2
    c,_,_,_ = np.linalg.lstsq(A,b, rcond=None)
    a, b_c, c_c = c
    cx = a; cy = b_c; r = np.sqrt(c_c + cx**2 + cy**2)
    return {'cx':cx,'cy':cy,'r':r}

# -------------------------
# Symmetry detection
# -------------------------
def detect_rotational_symmetry(points, max_n=12, tol=4.0):
    # points: Nx2 array in (x,y)
    # center at centroid
    pts = np.array(points)
    center = pts.mean(axis=0)
    rel = pts - center
    tree = KDTree(rel)
    for n in range(max_n,1,-1):
        ok = True
        theta = 360.0/n
        R = lambda ang: np.array([[cos(radians(ang)),-sin(radians(ang))],[sin(radians(ang)),cos(radians(ang))]])
        for k in range(1,n):
            rot = rel @ R(k*theta).T
            dists, _ = tree.query(rot, k=1)
            if np.median(dists) > tol:
                ok = False
                break
        if ok:
            return {'rotational_order': n, 'center': tuple(center)}
    return {'rotational_order': 1, 'center': tuple(center)}

def detect_reflection_symmetry(points, tol=4.0, angle_res=2.0):
    pts = np.array(points)
    center = pts.mean(axis=0)
    rel = pts - center
    tree = KDTree(rel)
    for a in np.arange(0,180, angle_res):
        # reflect across line at angle a through origin: matrix
        ang = radians(a)
        R = np.array([[cos(2*ang), sin(2*ang)],[sin(2*ang), -cos(2*ang)]])
        reflected = rel @ R.T
        dists,_ = tree.query(reflected, k=1)
        if np.median(dists) < tol:
            return {'reflection_axis_angle_deg': a, 'center': tuple(center)}
    return None

# -------------------------
# Main flow
# -------------------------
def analyze_image(path, show=False):
    binary = read_and_preprocess(path, show=False)
    # detect dots using distance peaks on inverted strokes
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    dots = detect_dots(gray, binary)
    skel, G = skeleton_and_graph(binary, show=False)
    # collect all path points for symmetry / grid inference
    all_pts = []
    fits = []
    for u,v,data in G.edges(data=True):
        pixels = data['pixels']  # list of (x,y)
        pts = np.array(pixels)
        all_pts.append(pts)
        fit = fit_spline(pts, smooth=1e-1)
        fits.append(((u,v), fit))
    all_pts_flat = np.vstack(all_pts) if all_pts else np.zeros((0,2))
    # symmetry detection using nodes + sampled path points
    sampled = all_pts_flat[np.linspace(0, all_pts_flat.shape[0]-1, min(300, all_pts_flat.shape[0])).astype(int)] if all_pts_flat.shape[0]>0 else np.zeros((0,2))
    rot = detect_rotational_symmetry(sampled, max_n=12, tol=3.5)
    refl = detect_reflection_symmetry(sampled, tol=3.5)
    # grid spacing: if dots exist, compute nearest neighbor distance avg
    grid_spacing = None
    if dots.shape[0]>1:
        tree = KDTree(dots)
        dists, _ = tree.query(dots, k=2)
        nn = dists[:,1]
        grid_spacing = float(np.median(nn))
    result = {
        'dots': dots,
        'graph': G,
        'fits': fits,
        'rotational_symmetry': rot,
        'reflection_symmetry': refl,
        'grid_spacing': grid_spacing
    }
    if show:
        plt.figure(figsize=(6,6))
        # plot original
        img = cv2.imread(path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis('off')
        # overlay nodes
        if dots.size>0:
            plt.scatter(dots[:,0], dots[:,1], s=20, c='yellow')
        # overlay skeleton paths and control points
        for (u,v), fit in fits:
            pts = np.array(G.edges[u,v]['pixels'])
            plt.plot(pts[:,0], pts[:,1], linewidth=1)
            if fit['type']=='spline':
                tck = fit['tck']
                u_eval = np.linspace(0,1,200)
                x,y = splev(u_eval, tck)
                plt.plot(x,y, linewidth=1)
        plt.title(f"rot order {rot['rotational_order']} spacing {grid_spacing}")
        plt.show()
    return result

# -------------------------
# Example usage (CLI)
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kolam_from_image.py path/to/kolam.png")
        sys.exit(1)
    path = sys.argv[1]
    res = analyze_image(path, show=True)
    # Print readable equations for first few fits
    print("Grid spacing (median NN of dots):", res['grid_spacing'])
    print("Rotational symmetry:", res['rotational_symmetry'])
    print("Reflection symmetry:", res['reflection_symmetry'])
    cnt = 0
    for (u,v), fit in res['fits']:
        cnt += 1
        if cnt>10: break
        print("Edge", (u,v), "fit type:", fit['type'])
        if fit['type']=='line':
            print(f"  Line y = {fit['m']:.4f} x + {fit['c']:.4f}")
        elif fit['type']=='spline':
            print("  Spline: use scipy tck (parametric). Summary:", spline_to_equation(fit['tck'], name=f"Edge{u}_{v}"))
        elif fit['type']=='polyline':
            print("  Polyline with N pts:", fit['pts'].shape[0])
    print("Done.")

    import pickle
    with open("kolam_extraction.pkl", "wb") as f:
        pickle.dump(res, f)
    print("Saved kolam_extraction.pkl")