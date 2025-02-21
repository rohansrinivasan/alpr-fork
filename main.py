import cv2
import time
import json
import random
import datetime
import threading
import os
import subprocess
import numpy as np  # Add this import
from collections import deque

from color_detection import get_dominant_color
from ocr.fast_ocr import detect_truck_number, detect_text
from ocr.usdot_extractor import TruckInfoExtractor

DEBUG = True
DEBUG_NO_CAMERA = True if os.uname().sysname == "Darwin" else False

# RTSP stream URL
RTSP_URL = "rtsp://admin:admin123@192.168.1.16:554/cam/realmonitor?channel=1&subtype=0"

# Door open script
DOOR_OPEN_SCRIPT = "./scripts/gpio_toggle.sh"

# Motion detection parameters
MOTION_THRESHOLD = 1000  # Threshold for motion detection

truck_info_extractor = TruckInfoExtractor()

# Open RTSP stream
if not DEBUG_NO_CAMERA:
    cap = cv2.VideoCapture(RTSP_URL)
else:
    cap = None

# Initialize variables for motion detection
prev_frame = None
motion_detected = False

def detect_motion(frame):
    if DEBUG or DEBUG_NO_CAMERA:
        return True
    global prev_frame, motion_detected
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Initialize prev_frame if needed
    if prev_frame is None:
        prev_frame = gray
        return False
    
    # Calculate frame difference
    frame_diff = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Calculate changes
    motion_score = np.sum(thresh)
    motion_detected = motion_score > MOTION_THRESHOLD
    
    # Update previous frame
    prev_frame = gray
    
    return motion_detected

def get_vehicle_data(detected_number, image_path):
    # get dominant color
    dominant_color = get_dominant_color(image_path)
    print(f"Dominant color: {dominant_color}")

    if DEBUG:
        image_path = "./testimgs/test_img1.jpg"
    else:
        #TODO: get image from RTSP stream
        print('Implement STREAM')
        
    usdot_number, vin_number = truck_info_extractor.extract_info(image_path)
    print(f"Truck info: {usdot_number}, {vin_number}")

    if usdot_number is None:
        usdot_number = "N/A"

    if vin_number is None:
        vin_number = "N/A"

    timestamp = datetime.datetime.now().isoformat()
    test_data = [
        {
            "metadata": {
                "location_id": "Test_Location",
                "box_id": "Test_Camera",
                "datetime": timestamp
            },
            "vehicle_data": {
                "truck_number": detected_number,
                "usdot": usdot_number,
                "vin": vin_number,
                "truck_color": dominant_color,
                "trailers": [f"TRAILER{random.randint(100, 999)}" for _ in range(random.randint(0, 2))]
            },
            "model_version": "1.0"
        }
    ]

    # Save JSON output to a .txt file
    json_output_path = "./output/output.txt"
    with open(json_output_path, "w") as json_file:
        json.dump(test_data, json_file, indent=2)
    print(f"Output written to {json_output_path}")

if __name__ == "__main__":
    # Check camera only if we're not in debug mode
    if not DEBUG_NO_CAMERA and not cap.isOpened():
        print("Error: Could not open RTSP stream")
        exit(1)  # Note: using exit() instead of os.exit()

    allowed_vehicles = json.load(open("allowed_vehicles.json"))
    allowed_numbers = [vehicle['number'] for vehicle in allowed_vehicles['vehicles']]

    print("\n\n")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("🚗🚗 Starting vehicle detection...")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("\n")

    ticks = 0
    while ticks < 10:
        if DEBUG_NO_CAMERA:
            output_path = "./testimgs/test_img2.jpg"
            frame = cv2.imread(output_path)
            ret = True
        else:
            ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame")
            continue
        
        # Check for motion before processing
        if not DEBUG_NO_CAMERA and not detect_motion(frame):
            print("No motion detected")
            time.sleep(0.1)
            continue
            
        output_path = "./output/current_frame.jpg"
        if not DEBUG_NO_CAMERA:
            cv2.imwrite(output_path, frame)

        if DEBUG:
            output_path = "./testimgs/test_img2.jpg"
            frame = cv2.imread(output_path)

        # Detect text in the frame
        t1 = time.time()
        detected_text = detect_text(output_path)  # This will return any text found in the frame
        t2 = time.time()
        print(f"Time taken to detect text: {t2 - t1} seconds")
        
        if detected_text:
            # Check if any detected text matches allowed numbers
            result = detect_truck_number(output_path, allowed_numbers)
            
            if result:
                print(f"🚗 Vehicle found! Number: {result}")
                thread = threading.Thread(target=get_vehicle_data, 
                                        args=(result, output_path))
                thread.start()
            else:
                print("🚗 Vehicle not in allowed list")
        else:
            print("No text detected in frame")

        ticks += 1
        
        time.sleep(0.5)

# Release resources
if cap:
    cap.release()