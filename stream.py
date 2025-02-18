import cv2

# RTSP URL
rtsp_url = "rtsp://admin:admin123@192.168.1.21:554/cam/realmonitor?channel=1&subtype=0"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream")
else:
    # Read one frame
    import time

    t1 = time.time()
    ret, frame = cap.read()

    if ret:
        # Save the frame as an image
        cv2.imwrite("/home/kfsjetson/anpr/testimgs/captured_frame.jpg", frame)
    else:
        print("Error: Could not read frame")

    t2 = time.time()
    print(f'Time taken: {t2 - t1} seconds')

# Release the stream
cap.release()
