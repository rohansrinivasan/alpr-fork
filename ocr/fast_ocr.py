from paddleocr import PaddleOCR
import cv2
import time
import numpy as np

# 1. Initialize the OCR model (English or multi-language, numeric, etc.)
ocr = PaddleOCR(lang='en', debug=False, show_log=False, warmup=True, use_angle_cls=True, use_gpu=True)

# 1.5 ModelWarmup
ocr.ocr("./testimgs/us_dot_test.jpg")

# Pre-processing function
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

def detect_truck_number(img_path, allowed_numbers):
    # 3. Run OCR
    processed_image = preprocess_image(img_path)
    result = ocr.ocr(processed_image)

    # Process OCR results
    if not result or len(result) == 0:
        print(f"No text detected in {img_path}")
        return None
    
    if result is None or len(result) == 0 or result[0] is None or len(result[0]) == 0:
        return None

    for line in result[0]:  # result[0] contains the actual text detections
        text, confidence = line[1]  # Unpack text and confidence
        
        # 4. Clean up recognized text and convert to numeric only
        cleaned = ''.join(ch for ch in str(text) if ch.isdigit())
        
        # Skip if empty or too short
        if not cleaned or len(cleaned) < 4:
            continue

        # 5. Check in allow-list
        if cleaned in allowed_numbers:
            print(f"Match found: {cleaned}")
            return cleaned

    return None

if __name__ == "__main__":
    images = ["./testimgs/test_img1.jpg", "./testimgs/test_img2.jpg", "./testimgs/alpr_test1.jpg"]
    allowed_numbers = {"3037", "1024", "4567", "1101"}
    for image in images:
        t1 = time.time()
        result = detect_truck_number(image, allowed_numbers)
        t2 = time.time()
        print(f"Time taken: {t2 - t1} seconds")
        if result:
            print(f"Truck is allowed! \033[92m✓\033[0m {image.split('/')[-1]}")
        else:
            print(f"Truck not in allow list. \033[91m✗\033[0m {image.split('/')[-1]}")