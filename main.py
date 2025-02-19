import cv2
import time
from color_detection import get_dominant_color
import json
import random
import datetime
import threading

DEBUG = True

# RTSP stream URL
rtsp_url = "rtsp://admin:admin123@192.168.1.16:554/cam/realmonitor?channel=1&subtype=0"

from openalpr import Alpr
# Initialize OpenALPR (for USA plates, change as needed)
alpr = Alpr("us", "/etc/openalpr/openalpr.conf", "/usr/share/openalpr/runtime_data")

# from ocr.usdot_extractor import TruckInfoExtractor
# truck_info_extractor = TruckInfoExtractor()

from ocr.fast_ocr import detect_truck_number

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

    if DEBUG:
        image_path = "./testimgs/test_img2.jpg"
    else:
        #TODO: get image from RTSP stream
        print('Implement STREAM')
        
    usdot_number, vin_number = None, None #truck_info_extractor.extract_info(image_path)
    print(f"Truck info: {usdot_number}, {vin_number}")

    if usdot_number is None:
        usdot_number = "N/A"

    if vin_number is None:
        vin_number = "N/A"

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
    allowed_vehicles = json.load(open("allowed_vehicles.json"))
    allowed_numbers = [vehicle['number'] for vehicle in allowed_vehicles['vehicles']]
    while total_seconds < 10:
        # Capture frame
        ret, frame = cap.read()

        if ret:
            # Save frame to output folder and get the path
            output_path = "./output/current_frame.jpg"
            cv2.imwrite(output_path, frame)

            if DEBUG:
                # use test image instead of current frame
                output_path = "./testimgs/alpr_test1.jpg"

            # Run ALPR directly on current frame
            results = alpr.recognize_file(output_path)

            found = False

            # Check results
            for result in results['results']:
                plate_text = result['plate']
                confidence = result['confidence']
                
                print(f"Detected Plate: {plate_text} | Confidence: {confidence:.2f}%")

                result = detect_truck_number(output_path, allowed_numbers)
                if result:
                    found = True
                    print(f"🚗 Vehicle found! Number: {result}")
                    # thread = threading.Thread(target=get_vehicle_data, args=(plate_text, output_path))
                    # thread.start()
                    break
            if not found:
                print("🚗 Vehicle not in allowed list")

        else:
            print("Error: Could not read frame")

        if DEBUG:
            total_seconds += 1
        time.sleep(1)  # Wait for 1 seconds before capturing the next frame

# Release resources
cap.release()
alpr.unload()