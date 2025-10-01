import os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

def preprocess_kolam(image_path, target_size=(300, 300), threshold=128):
    # Load image
    img = Image.open(image_path).convert("L")  # grayscale
    
    # --- 1. Gentle denoising (removes small specks, keeps lines sharp) ---
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    # --- 2. Binarize image ---
    img = img.point(lambda p: 255 if p > threshold else 0, mode="1").convert("L")
    
    # --- 3. Ensure background is black & lines are white ---
    bg_pixel = img.getpixel((0, 0))
    if bg_pixel > 128:
        img = ImageOps.invert(img)
    
    # --- 5. Resize ---
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # --- 6. Improve contrast while preserving line info ---
    img = ImageEnhance.Contrast(img).enhance(2.0)
    
    # Optional: slight brightness adjust if lines look too faint
    img = ImageEnhance.Brightness(img).enhance(0.8)
    
    # --- Save with new name ---
    dx = os.path.dirname(image_path)
    base, ext = os.path.splitext(os.path.basename(image_path))  # safer split
    
    # remove original image (optional, careful!)
    if os.path.exists(image_path):
        os.remove(image_path)
    
    out_path = os.path.join(dx, f"{base}_preprocessed{ext}")
    img.save(out_path)

    return out_path