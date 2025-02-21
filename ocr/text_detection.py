import cv2
import numpy as np

def detect_text(image):
    """
    Detect text in an image using OpenCV's EAST text detector
    Returns: True if any text is detected, False otherwise
    """
    # Store original image dimensions
    orig_h, orig_w = image.shape[:2]
    
    # Set parameters
    conf_threshold = 0.5
    nms_threshold = 0.3
    width = 320  # smaller size for faster processing
    height = 320
    
    # Calculate ratio for scaling back to original size
    ratio_w = orig_w / float(width)
    ratio_h = orig_h / float(height)
    
    # Prepare image for the model
    blob = cv2.dnn.blobFromImage(image, 1.0, (width, height),
                                (123.68, 116.78, 103.94), True, False)
    
    try:
        # Load the pre-trained EAST text detector
        net = cv2.dnn.readNet("frozen_east_text_detection.pb")
        
        # Set the blob as input
        net.setInput(blob)
        
        # Define output layers
        output_layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        scores, geometry = net.forward(output_layers)
        
        # Get number of rows and columns in scores
        rows = scores.shape[2]
        cols = scores.shape[3]
        
        # Initialize rectangles and confidences lists
        rectangles = []
        confidences = []
        
        # Loop over each cell in the scores matrix
        for y in range(rows):
            scores_data = scores[0, 0, y]
            for x in range(cols):
                if scores_data[x] < conf_threshold:
                    continue
                
                # Found text with confidence above threshold
                return True
                
        return False
        
    except Exception as e:
        print(f"Error in text detection: {e}")
        # If model file is missing or other error, return False
        return False 