from ultralytics import YOLO
import cv2
import numpy as np
import json
import datetime
from collections import Counter

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load allowed vehicles list
with open("allowed_vehicles.json", "r") as f:
    allowed_vehicles = json.load(f)

# Open webcam (0 = default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Keep track of detected vehicles
tracked_vehicles = set()
frame_empty = True  # To detect when a new vehicle arrives

# Function to get the dominant color using K-Means clustering
def get_dominant_color(image, k=4):
    if image is None or image.size == 0:
        return "Unknown"

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # Focus on the central region (avoid background influence)
    center_x, center_y = w // 2, h // 2
    cropped = image[center_y - h//4:center_y + h//4, center_x - w//4:center_x + w//4]
    pixels = cropped.reshape((-1, 3)).astype(np.float32)

    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Find the most common cluster
    label_counts = Counter(labels.flatten())
    dominant_color = centers[max(label_counts, key=label_counts.get)]

    return get_color_name(dominant_color)

# Function to map RGB to a general color name
def get_color_name(rgb):
    colors = {
        "Red": (220, 20, 60),
        "Green": (34, 139, 34),
        "Blue": (30, 144, 255),
        "Black": (0, 0, 0),
        "White": (255, 255, 255),
        "Yellow": (255, 255, 0),
        "Cyan": (0, 255, 255),
        "Magenta": (255, 0, 255),
        "Gray": (128, 128, 128),
        "Orange": (255, 140, 0),
        "Brown": (139, 69, 19)
    }

    min_distance = float("inf")
    closest_color = "Unknown"
    for color_name, ref_rgb in colors.items():
        distance = np.linalg.norm(np.array(rgb) - np.array(ref_rgb))
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    # Ignore white/gray if another strong color is detected
    if closest_color in ["White", "Gray"]:
        second_choice = sorted(colors.keys(), key=lambda c: np.linalg.norm(np.array(rgb) - np.array(colors[c])))[1]
        if second_choice not in ["White", "Gray"]:
            return second_choice

    return closest_color

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame.")
        break

    # Run YOLOv8 inference
    results = model(frame)

    current_vehicles = set()
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0].item())
            vehicle_name = model.names[cls]

            # Extract vehicle type and crop the region
            vehicle_crop = frame[y1:y2, x1:x2]
            vehicle_color = get_dominant_color(vehicle_crop)

            # Check if this vehicle type and color is in the allowed list
            if any(v["type"] in vehicle_name and v["color"] == vehicle_color for v in allowed_vehicles):
                vehicle_id = f"{vehicle_color} {vehicle_name}"
                current_vehicles.add(vehicle_id)

                # Only log if it's a new vehicle
                if vehicle_id not in tracked_vehicles:
                    tracked_vehicles.add(vehicle_id)

                    # Generate JSON output
                    timestamp = datetime.datetime.now().isoformat()
                    json_output = [{
                        "metadata": {
                            "location_id": "Test_Location",
                            "box_id": "Test_Camera",
                            "datetime": timestamp
                        },
                        "vehicle_data": {
                            "plate_no": "XYZ123",
                            "usdot": "1234567",
                            "vin": "ABCDEFG123456789",
                            "truck_color": vehicle_color,
                            "trailers": []
                        },
                        "model_version": "1.2"
                    }]

                    # Append to output file
                    with open("output.txt", "a") as json_file:
                        json.dump(json_output, json_file, indent=2)
                        json_file.write("\n")
                    print(f"Added {vehicle_id} to output.txt")

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{vehicle_color} {vehicle_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # If no vehicles detected, reset tracking
    if not current_vehicles and not frame_empty:
        print("Frame empty, resetting tracked vehicles...")
        tracked_vehicles.clear()
        frame_empty = True
    elif current_vehicles:
        frame_empty = False

    # Show the live feed
    cv2.imshow("Live Vehicle Detection with Color", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


# from ultralytics import YOLO
# import cv2
# import numpy as np
# from collections import Counter

# # Load YOLOv8 model
# model = YOLO("yolov8n.pt")  # Use a better-trained model if available

# # Open webcam (0 = default webcam)
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Function to get the dominant color using K-Means clustering
# def get_dominant_color(image, k=4):
#     if image is None or image.size == 0:
#         return "Unknown"

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     h, w, _ = image.shape

#     # Focus on the central region (ignore edges and reflections)
#     center_x, center_y = w // 2, h // 2
#     cropped = image[center_y - h//4:center_y + h//4, center_x - w//4:center_x + w//4]

#     pixels = cropped.reshape((-1, 3)).astype(np.float32)

#     # Apply K-means clustering
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

#     # Find the most common cluster
#     label_counts = Counter(labels.flatten())
#     dominant_color = centers[max(label_counts, key=label_counts.get)]

#     return get_color_name(dominant_color)

# # Function to map RGB to a general color name
# def get_color_name(rgb):
#     colors = {
#         "Red": (220, 20, 60),
#         "Green": (34, 139, 34),
#         "Blue": (30, 144, 255),
#         "Black": (0, 0, 0),
#         "White": (255, 255, 255),
#         "Yellow": (255, 255, 0),
#         "Cyan": (0, 255, 255),
#         "Magenta": (255, 0, 255),
#         "Gray": (128, 128, 128),
#         "Orange": (255, 140, 0),
#         "Brown": (139, 69, 19)
#     }

#     min_distance = float("inf")
#     closest_color = "Unknown"
#     for color_name, ref_rgb in colors.items():
#         distance = np.linalg.norm(np.array(rgb) - np.array(ref_rgb))
#         if distance < min_distance:
#             min_distance = distance
#             closest_color = color_name

#     # Ignore white/gray if a stronger color exists
#     if closest_color in ["White", "Gray"]:
#         second_choice = sorted(colors.keys(), key=lambda c: np.linalg.norm(np.array(rgb) - np.array(colors[c])))[1]
#         if second_choice not in ["White", "Gray"]:
#             return second_choice

#     return closest_color

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not capture frame.")
#         break

#     # Run YOLOv8 inference
#     results = model(frame)

#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
#             cls = int(box.cls[0].item())  # Get class ID
#             confidence = box.conf[0].item()  # Get confidence score
#             vehicle_name = model.names[cls]  # Get class label

#             # Extract the detected vehicle region
#             vehicle_crop = frame[y1:y2, x1:x2]

#             # Get dominant color of the cropped vehicle
#             vehicle_color = get_dominant_color(vehicle_crop)

#             # Draw bounding box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             # Overlay detected color and vehicle type
#             text = f"{vehicle_color} {vehicle_name}"
#             cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     # Show the live feed
#     cv2.imshow("Live Vehicle Detection with Color", frame)

#     # Break loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()


# from ultralytics import YOLO
# import cv2
# import json
# import datetime

# # Load YOLOv8 model (pre-trained on vehicles)
# model = YOLO("yolov8n.pt")  # You can replace with a model trained on vehicle colors if available

# # Open webcam (0 = default webcam)
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open webcam.")
# else:
#     ret, frame = cap.read()
#     if ret:
#         # Save the screenshot
#         screenshot_path = "screenshot.jpg"
#         cv2.imwrite(screenshot_path, frame)
#         print(f"Screenshot saved as {screenshot_path}")

#         # Run YOLOv8 inference on the frame
#         results = model(frame)

#         detected_vehicles = []
#         for result in results:
#             for box in result.boxes:
#                 cls = int(box.cls[0].item())  # Get class ID
#                 confidence = box.conf[0].item()  # Get confidence score
#                 vehicle_name = model.names[cls]  # Get class label

#                 # Extract vehicle color from class name if available
#                 vehicle_color = vehicle_name.split(" ")[0] if " " in vehicle_name else "Unknown"

#                 detected_vehicles.append({
#                     "plate_no": "XYZ123",  # Placeholder, replace with ANPR output
#                     "usdot": "1234567",
#                     "vin": "ABCDEFG123456789",
#                     "truck_color": vehicle_color,
#                     "trailers": []
#                 })

#         # Generate JSON output
#         timestamp = datetime.datetime.now().isoformat()
#         json_output = [{
#             "metadata": {
#                 "location_id": "Test_Location",
#                 "box_id": "Test_Webcam",
#                 "datetime": timestamp
#             },
#             "vehicle_data": detected_vehicles[0] if detected_vehicles else {},
#             "model_version": "1.2"
#         }]

#         # Save JSON output
#         json_output_path = "output.txt"
#         with open(json_output_path, "w") as json_file:
#             json.dump(json_output, json_file, indent=2)
#         print(f"Output written to {json_output_path}")

#     else:
#         print("Error: Could not capture frame.")

# # Release the capture
# cap.release()
