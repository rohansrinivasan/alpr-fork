import cv2
import json
import random
import datetime
import numpy as np
from collections import Counter
from color_detection import get_dominant_color
# Replace with your camera's RTSP or HTTP stream URL
RTSP_URL = "rtsp://admin:admin123@192.168.1.16:554/cam/realmonitor?channel=1&subtype=0"
TRUCK_IMG_PATH = "/home/kfsjetson/code/alpr/testimgs/alpr_test1.jpg"

def main():
    # Open video capture
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("Error: Could not open camera stream.")
        return

    ret, frame = cap.read()
    if ret:
        # Save the screenshot
        screenshot_path = "./output/screenshot.jpg"
        cv2.imwrite(screenshot_path, frame)
        print(f"Screenshot saved as {screenshot_path}")

        # Detect truck color
        truck_color = get_dominant_color(TRUCK_IMG_PATH)
        print(f'Truck color is {truck_color}')

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
                    "plate_no": f"XYZ{random.randint(100, 999)}",
                    "usdot": str(random.randint(1000000, 9999999)),
                    "vin": f"VIN{random.randint(100000, 999999)}",
                    "truck_color": truck_color,
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
    else:
        print("Error: Could not capture frame.")

    # Release the capture
    cap.release()

if __name__ == "__main__":
    main()