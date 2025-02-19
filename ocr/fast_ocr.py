from paddleocr import PaddleOCR
import cv2
import time

# 1. Initialize the OCR model (English or multi-language, numeric, etc.)
ocr = PaddleOCR(lang='en')  # Set use_gpu=True if GPU is available

# 2. Allowed list
allowed_numbers = {"3037", "1024", "4567", "1101"}

# Pre-processing function
def preprocess_image(image_path):
    img = cv2.imread(image_path)

    # Convert to grayscale (faster processing)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Increase contrast using histogram equalization
    enhanced = cv2.equalizeHist(gray)

    # Apply adaptive thresholding to make text stand out
    return enhanced

def detect_truck_number(img_path):
    # 3. Run OCR
    processed_image = preprocess_image(img_path)
    result = ocr.ocr(processed_image)
    print(result)
    
    # sample result:
    '''
    [[[[[460.0, 294.0], [542.0, 294.0], [542.0, 341.0], [460.0, 341.0]], ('3037', 0.9988995790481567)], [[[641.0, 555.0], [740.0, 549.0], [743.0, 593.0], [644.0, 598.0]], ('08KS3W', 0.9172372221946716)]]]
    '''

    # result is typically a list of lists: [ [ [box], (text, confidence) ], ... ]
    for line in result[0]:  # Access first page's results
        if line is not None:
            text = line[1][0]  # Get text from tuple (text, confidence)
            
            # Clean up recognized text
            cleaned = ''.join(ch for ch in text if ch.isalnum())
            
            # Check in allow-list
            if cleaned in allowed_numbers:
                return True  # Found a match

    return False

# Example usage:
t1 = time.time()
result = detect_truck_number("./testimgs/test_img2.jpg")
t2 = time.time()
print(f"Time taken: {t2 - t1} seconds")
if result:
    print("Truck is allowed!")
else:
    print("Truck not in allow list.")
