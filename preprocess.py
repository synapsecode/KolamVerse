from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np

def preprocess_image(image_path, output_path=None, target_size=(128, 128), threshold=128):
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
    
    # --- 4. Thicken lines slightly (better connectivity for kolam2csv) ---
    img = img.filter(ImageFilter.MaxFilter(size=3))
    
    # --- 5. Resize ---
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # --- 6. Improve contrast while preserving line info ---
    img = ImageEnhance.Contrast(img).enhance(2.0)
    
    # Optional: slight brightness adjust if lines look too faint
    img = ImageEnhance.Brightness(img).enhance(0.8)
    
    # Save if needed
    if output_path:
        img.save(output_path)
    
    return img

# Example
processed = preprocess_image("a.png", "output.jpg")
processed.show()
