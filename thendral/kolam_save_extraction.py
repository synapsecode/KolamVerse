# kolam_save_extraction.py
import sys
import pickle
import argparse

# change this if your analyzer module has a different filename
# it should define analyze_image(path, show=False) and return the result dict
import kolam_from_image as kfi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Input kolam image file (png/jpg)")
    ap.add_argument("--out", "-o", default="kolam_extraction.pkl", help="Output pickle filename")
    args = ap.parse_args()

    print("Analyzing image:", args.image)
    res = kfi.analyze_image(args.image, show=False)
    # optionally attach the image path
    res['source_image'] = args.image

    # save with pickle (contains numpy arrays, tck objects etc)
    with open(args.out, "wb") as f:
        pickle.dump(res, f)

    print("Saved extraction to", args.out)
    # summary
    print("Grid spacing:", res.get('grid_spacing'))
    print("rotational_symmetry:", res.get('rotational_symmetry'))
    print("reflection_symmetry:", res.get('reflection_symmetry'))
    print("Number of edges:", len(res['fits']))

if __name__ == "__main__":
    main()