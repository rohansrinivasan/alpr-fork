import cv2
import numpy as np
import os

DEBUG_DIR = "./debug/ocr"
if os.path.exists(DEBUG_DIR):
    for file in os.listdir(DEBUG_DIR):
        os.remove(os.path.join(DEBUG_DIR, file))
os.makedirs(DEBUG_DIR, exist_ok=True)

def preprocess_image(image_path, scale_down=1.0):
    img = cv2.imread(image_path)

    # Convert to grayscale (faster processing)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # scale down image
    if scale_down != 1.0:
        gray = cv2.resize(gray, (0, 0), fx=scale_down, fy=scale_down, interpolation=cv2.INTER_AREA)

    # # Increase contrast using CLAHE (more advanced than simple histogram equalization)
    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    # enhanced = clahe.apply(gray)

    # Optional: Morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    return cleaned

if __name__ == "__main__":
    # image_path = "./testimgs/us_dot_test.jpg"
    image_path = "./testimgs/test_img1.jpg"
    processed_image = preprocess_image(image_path)
    cv2.imwrite(os.path.join(DEBUG_DIR, "4_processed_image.jpg"), processed_image)
