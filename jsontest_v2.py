import cv2
import json
import random
import datetime
import numpy as np
from collections import Counter

# Replace with your camera's RTSP or HTTP stream URL
rtsp_url = "rtsp://admin:admin123@192.168.1.16:554/cam/realmonitor?channel=1&subtype=0"
TRUCK_IMG_PATH = "/home/kfsjetson/code/alpr/testimgs/alpr_test1.jpg"

def get_dominant_color(image_path, k=3, downscale_factor=0.5):
    # Load and downscale image
    image = cv2.imread(image_path)
    if image is None:
        return "Unknown"
    
    # Downscale image for faster processing
    height, width = image.shape[:2]
    new_height, new_width = int(height * downscale_factor), int(width * downscale_factor)
    image = cv2.resize(image, (new_width, new_height))
    
    # Create a mask focusing on central area
    height, width = image.shape[:2]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Create mask for vehicle body - common color area
    center_y_body = int(height * 0.5)
    center_x = width // 2
    ellipse_width = int(width * 0.5)
    ellipse_height = int(height * 0.35)
    
    # Body mask (central area)
    cv2.ellipse(mask, (center_x, center_y_body), (ellipse_width, ellipse_height), 
                0, 0, 360, 255, -1)
    
    # Apply mask
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    
    # Convert to HSV color space - use BGR first for direct RGB analysis
    bgr_pixels = masked_img[np.where(mask > 0)]
    hsv_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
    
    # Filter out black pixels (masked out)
    non_zero_indices = np.where(mask > 0)
    non_zero_pixels = hsv_img[non_zero_indices]
    
    # Skip processing if not enough pixels
    if len(non_zero_pixels) < 50:
        return "Unknown"
    
    # Get average RGB values directly - useful for color verification
    avg_b = np.mean(bgr_pixels[:, 0]).astype(int) 
    avg_g = np.mean(bgr_pixels[:, 1]).astype(int)
    avg_r = np.mean(bgr_pixels[:, 2]).astype(int)
    
    # Reshape for k-means - use fewer samples for faster processing
    max_samples = min(5000, len(non_zero_pixels))
    sample_indices = np.random.choice(len(non_zero_pixels), max_samples, replace=False)
    pixels = non_zero_pixels[sample_indices].reshape((-1, 3)).astype(np.float32)
    
    # Apply K-means clustering with fewer iterations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    
    # Find frequency of each cluster
    label_counts = Counter(labels.flatten())
    
    # Calculate basic hue histogram for color analysis - but use fewer bins
    hue_bins = np.zeros(18)  # 10 degree bins instead of 1 degree
    
    # Only consider pixels with reasonable saturation and value
    for pixel in pixels:
        h, s, v = pixel
        if s > 20 and v > 40:
            hue_bins[int(h/10)] += 1
    
    # Calculate range totals with simplified ranges
    green_range = np.sum(hue_bins[6:10])  # Green: 60-100
    blue_range = np.sum(hue_bins[10:14])  # Blue: 100-140
    red_range = np.sum(hue_bins[0:1]) + np.sum(hue_bins[17:18])
    yellow_range = np.sum(hue_bins[2:4])
    
    # Quick check for dominant colors based on RGB averages
    # This helps catch reds that might be missed in HSV
    if avg_r > avg_g + 50 and avg_r > avg_b + 50 and avg_r > 120:
        return "Red"  # Strong red signature
        
    # Get clusters for analysis
    cluster_data = []
    
    for i, center in enumerate(centers):
        h, s, v = center
        count = label_counts.get(i, 0)
        percentage = (count / len(labels)) * 100
        
        # Convert HSV center to BGR and RGB for comparison
        bgr = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_HSV2BGR)[0][0]
        rgb = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2RGB)[0][0]
        r, g, b = rgb
        
        # Skip very dark or very light unsaturated colors
        if (v < 40) or (s < 25 and v > 200):
            continue
            
        # Better blue vs. green classification
        if b > g + 20 and b > r + 20:
            color_classifier = "strong blue"
        elif g > b + 20 and g > r:
            color_classifier = "strong green"
        elif r > g + 30 and r > b + 30:
            color_classifier = "strong red"
        elif 100 <= h <= 140 and b >= g:
            color_classifier = "blue-ish"
        elif 60 <= h < 100 and g >= b:
            color_classifier = "green-ish"
        elif (h < 10 or h > 170) and r > max(g, b):
            color_classifier = "red-ish"
        else:
            color_classifier = "unknown"
        
        # Calculate a weighted score - boost scores for primary colors
        count_weight = count / len(labels)
        saturation_weight = s / 255.0
        value_weight = min(v / 255.0, 0.9)
        
        # Calculate color distinctiveness differently for each color type
        if "blue" in color_classifier:
            color_distinctiveness = (b - max(r, g)) / 255.0  # Blue distinctiveness
        elif "green" in color_classifier:
            color_distinctiveness = (g - max(r, b)) / 255.0  # Green distinctiveness
        elif "red" in color_classifier:
            color_distinctiveness = (r - max(g, b)) / 255.0  # Red distinctiveness
        else:
            color_distinctiveness = 0.1  # Default for unknown
            
        score = (count_weight * 0.5 + 
                saturation_weight * 0.2 + 
                value_weight * 0.1 + 
                color_distinctiveness * 0.2)
        
        cluster_data.append({
            'hsv': center,
            'rgb': rgb,
            'count': count,
            'percentage': percentage,
            'score': score,
            'h': h,
            's': s,
            'v': v,
            'r': r,
            'g': g,
            'b': b,
            'classifier': color_classifier
        })
    
    # If no clusters passed filtering
    if not cluster_data:
        return "Unknown"
    
    # Sort clusters by score
    sorted_clusters = sorted(cluster_data, key=lambda x: x['score'], reverse=True)
    
    # Get the top scoring cluster
    top_cluster = sorted_clusters[0]
    h, s, v = top_cluster['hsv']
    r, g, b = top_cluster['rgb']
    
    # Special handling for certain colors that are harder to classify
    
    # RED detection fix: Check for high red component
    if r > 150 and r > g + 40 and r > b + 40:
        return "Red"
    
    # Special case for Red that wraps in HSV space
    if (h < 10 or h > 170) and s > 50 and r > max(g, b):
        return "Red"
    
    # Improved BLUE detection - check if avg blue is significantly higher
    if b > g + 25 and b > r and "blue" in top_cluster['classifier']:
        return "Blue"
    
    # BLUE-GREEN boundary case with special handling
    if (90 <= h <= 110) or (abs(int(b) - int(g)) < 30):
        # Trust RGB values for blue-green distinction
        if int(b) > int(g) + 15:
            return "Blue"
        elif int(g) > int(b) + 15:
            return "Green"
            
        # If RGB is inconclusive, use histogram
        if blue_range > green_range * 1.5:
            return "Blue"
        elif green_range > blue_range * 1.5:
            return "Green"
        
        # Edge case: check the 2nd highest cluster for blue-green distinction
        if len(sorted_clusters) > 1:
            second_cluster = sorted_clusters[1]
            if "blue" in second_cluster['classifier'] and second_cluster['b'] > second_cluster['g']:
                return "Blue"
            elif "green" in second_cluster['classifier'] and second_cluster['g'] > second_cluster['b']:
                return "Green"
    
    # Run the refined color classification for the final decision
    final_color = refined_vehicle_color_classification(h, s, v, r, g, b)
    return final_color

def refined_vehicle_color_classification(h, s, v, r, g, b):
    """Improved vehicle color classification combining HSV and RGB analysis"""
    
    # Very low saturation = grayscale
    if s < 25:
        if v < 80:
            return "Black"
        elif v > 180:
            return "White"
        else:
            return "Gray/Silver"
    
    # RED detection (HSV has trouble with red because it wraps around 0/180)
    if ((h < 10 or h > 165) and s > 50) or (r > 150 and r > g*1.5 and r > b*1.5):
        return "Red"
    
    # BLUE detection improved with RGB check
    if (100 <= h <= 150 and b > g) or (b > g + 15 and b > r):
        # Stronger blue has higher difference between b and g
        if b > g + 30 or b > 150:
            return "Blue"
        else:
            # Metallic blues often have lower saturation
            if s < 70:
                return "Light Blue"
            else:
                return "Blue"
    
    # GREEN detection with more precise boundaries
    if (60 <= h < 100 and g > b) or (g > b + 15 and g > r):
        if g > b + 30 or g > 150:
            return "Green" 
        else:
            if s < 70:
                return "Olive Green"
            else:
                return "Green"
    
    # Yellow range
    if 20 <= h <= 40 and s > 50:
        if s > 100 and v > 150:
            return "Yellow"
        else:
            return "Gold/Beige"
            
    # Orange is between red and yellow
    if 10 <= h < 20 and s > 50:
        return "Orange"
    
    # Teal/Cyan range
    if 80 <= h < 95 and b > r:
        return "Teal"
    
    # Purple range
    if 140 < h < 165:
        return "Purple"
    
    # Fallback based on RGB ratio when in doubt
    if r > max(g, b) + 30:
        return "Red"
    elif b > g + 15:
        return "Blue"
    elif g > b + 15:
        return "Green"
    
    # Final fallback
    if s < 60 and v > 100:
        return "Metallic/Silver"
    
    return "Unknown"

def main():
    # Open video capture
    cap = cv2.VideoCapture(rtsp_url)

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