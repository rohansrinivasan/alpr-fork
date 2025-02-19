import re
import numpy as np
import cv2

from ocr.fast_ocr import run_ocr

def preprocess_image(image_path, resize_width=None):
    """
    Preprocess image to improve OCR performance
    
    Args:
        image_path (str): Path to the image file
        resize_width (int, optional): Width to resize image to. If None, no resizing is performed
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize image if specified
    if resize_width is not None:
        height, width = img.shape
        if width > resize_width:
            scale = resize_width / width
            new_width = resize_width
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
    
    # Enhance contrast
    img = cv2.equalizeHist(img)
    
    # Denoise
    img = cv2.fastNlMeansDenoising(img)
    
    return img

class TruckInfoExtractor:
    def __init__(self, use_gpu=True, cpu_threads=4):
        """
        Initialize OCR once and reuse it for multiple images
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration
            cpu_threads (int): Number of CPU threads to use
        """
    
    def extract_info(self, image_path, resize_width=None):
        """
        Extract USDOT and VIN numbers from an image of a truck.
        
        Args:
            image_path (str): Path to the image file
            resize_width (int, optional): Width to resize image to. If None, no resizing is performed
            
        Returns:
            tuple: (usdot_number, vin_number) - Both could be None if not found
        """
        # Preprocess the image
        img = preprocess_image(image_path, resize_width)
        
        # Initialize return values
        usdot_number = None
        vin_number = None
        
        # Perform OCR on the preprocessed image
        try:
            result = run_ocr(img)
            
            if result and len(result) > 0:
                # Convert results to a list of text strings
                texts = [line[1][0] for line in result[0] if line]
                
                # Look for USDOT and VIN in the detected texts
                for i, text in enumerate(texts):
                    # Search for USDOT
                    if 'USDOT' in text.upper():
                        # Check if the number is in the same text
                        number_match = re.search(r'\d{5,7}', text)
                        if number_match:
                            usdot_number = number_match.group(0)
                        # If not, check the next text item if available
                        elif i + 1 < len(texts):
                            next_text = texts[i + 1]
                            number_match = re.search(r'\d{5,7}', next_text)
                            if number_match:
                                usdot_number = number_match.group(0)
                    
                    # Search for VIN
                    if 'VIN' in text.upper():
                        # Look for VIN pattern in current and next text
                        vin_pattern = r'[A-Z0-9]{6,8}'  # Adjust pattern based on your needs
                        vin_match = re.search(vin_pattern, text.upper())
                        if vin_match:
                            vin_number = vin_match.group(0)
                        # Check next text if VIN not found in current text
                        elif i + 1 < len(texts):
                            next_text = texts[i + 1]
                            vin_match = re.search(vin_pattern, next_text.upper())
                            if vin_match:
                                vin_number = vin_match.group(0)
        
        except Exception as e:
            print(f"Error processing image: {e}")
            return None, None
        
        return usdot_number, vin_number

def main():
    """
    Test function to demonstrate truck info extraction
    """
    # Create extractor instance once
    extractor = TruckInfoExtractor(
        use_gpu=True,
        cpu_threads=4
    )
    
    # Test the function
    image_path = "./testimgs/test_img1.jpg"
    
    import time
    
    # Test without resizing
    start_time = time.time()
    usdot_number, vin_number = extractor.extract_info(image_path, resize_width=None)
    processing_time = time.time() - start_time
    
    print("\nWithout resizing:")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Found USDOT number: {usdot_number if usdot_number else 'Not found'}")
    print(f"Found VIN number: {vin_number if vin_number else 'Not found'}")
    
    # Test with resizing
    start_time = time.time()
    usdot_number, vin_number = extractor.extract_info(image_path, resize_width=1280)
    processing_time = time.time() - start_time
    
    print("\nWith resizing to 1280px width:")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Found USDOT number: {usdot_number if usdot_number else 'Not found'}")
    print(f"Found VIN number: {vin_number if vin_number else 'Not found'}")

if __name__ == "__main__":
    main() 