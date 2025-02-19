from PIL import Image
import cv2
from paddleocr import PaddleOCR
import numpy as np

# Initialize PaddleOCR
ocr = PaddleOCR(lang='en')

# Using PIL
image = Image.open('./testimgs/test_img1.jpg').convert('RGB')
image_np = np.array(image)

dummy_image = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
result = ocr.ocr(dummy_image)

import time
start_time = time.time()
result = ocr.ocr(image_np)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print(result)

start_time = time.time()
result = ocr.ocr(image_np)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")