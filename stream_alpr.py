import cv2
import time
from openalpr import Alpr

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

while total_seconds < 10:
    # Capture frame
    ret, frame = cap.read()

    if ret:
        # Convert to JPEG format and get byte array
        _, encoded_image = cv2.imencode('.jpg', frame)  # Encode as JPEG
        image_bytes = encoded_image.tobytes()  # Convert to bytes

        # Run ALPR directly on the byte array
        results = alpr.recognize_array(image_bytes)

        confidence = 0

        # Check results
        for result in results['results']:
            plate_text = result['plate']
            confidence = result['confidence']
            
            print(f"Detected Plate: {plate_text} | Confidence: {confidence:.2f}%")

            if confidence >= 80:
                print("🚗 Vehicle found!")
                break
        
        if confidence < 80:
            print("🚗 Vehicle not found!")

    else:
        print("Error: Could not read frame")

    total_seconds += 1
    time.sleep(1)  # Wait for 1 seconds before capturing the next frame

# Release resources
cap.release()
alpr.unload()