import cv2
import time
from openalpr import Alpr
from color_detection import get_dominant_color
import json
import random
import datetime
import threading

DEBUG = True

# RTSP stream URL
rtsp_url = "rtsp://admin:admin123@192.168.1.16:554/cam/realmonitor?channel=1&subtype=0"

# Initialize OpenALPR (for USA plates, change as needed)
alpr = Alpr("us", "/etc/openalpr/openalpr.conf", "/usr/share/openalpr/runtime_data")

if not alpr.is_loaded():
    print("Error: OpenALPR failed to load")
    exit()

# Open RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream")
    exit()

total_seconds = 0

def get_vehicle_data(plate_text, image_path):
    # get dominant color
    dominant_color = get_dominant_color(image_path)
    print(f"Dominant color: {dominant_color}")

    # Generate test JSON data
    timestamp = datetime.datetime.now().isoformat()
    test_data = [
        {
            "metadata": {
                "location_id": "Test_Location",
                "box_id": "Test_Camera",
                "datetime": timestamp
            },
            "vehicle_data": {
                "plate_no": plate_text,
                "usdot": str(random.randint(1000000, 9999999)),
                "vin": f"VIN{random.randint(100000, 999999)}",
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

while total_seconds < 10:
    # Capture frame
    ret, frame = cap.read()

    if ret:
        # Save frame to output folder and get the path
        output_path = "./output/current_frame.jpg"
        cv2.imwrite(output_path, frame)

        if DEBUG:
            # use test image instead of current frame
            output_path = "./testimgs/test_img1.jpg"

        # Run ALPR directly on current frame
        results = alpr.recognize_file(output_path)

        confidence = 0
        plate_text = ""

        # Check results
        for result in results['results']:
            plate_text = result['plate']
            confidence = result['confidence']
            
            print(f"Detected Plate: {plate_text} | Confidence: {confidence:.2f}%")

            if confidence >= 80:
                print("🚗 Vehicle found!")
                thread = threading.Thread(target=get_vehicle_data, args=(plate_text, output_path))
                thread.start()
                break
        
        if confidence < 80:
            print("🚗 Vehicle not found!")

    else:
        print("Error: Could not read frame")

    if DEBUG:
        total_seconds += 1
    time.sleep(1)  # Wait for 1 seconds before capturing the next frame

# Release resources
cap.release()
alpr.unload()