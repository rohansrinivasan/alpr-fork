import cv2
import json
import random
import datetime
import numpy as np
from collections import Counter

# Replace with your camera's RTSP or HTTP stream URL
rtsp_url = "rtsp://admin:admin123@192.168.1.16:554/cam/realmonitor?channel=1&subtype=0"
TRUCK_IMG_PATH = "./testimgs/alpr_test1.jpg"  # Ensure an image of a truck is in the same directory

def get_dominant_color(image_path, k=5):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the truck image.")
        return "Unknown"
    
    # Debug - display image dimensions
    print(f"Image shape: {image.shape}")
    
    # Create a mask focusing on central area
    height, width = image.shape[:2]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Create two masks - one for body, one for hood/roof - common color areas
    center_y_body = int(height * 0.5)
    center_x = width // 2
    ellipse_width = int(width * 0.5)
    ellipse_height = int(height * 0.35)
    
    # Body mask (central area)
    cv2.ellipse(mask, (center_x, center_y_body), (ellipse_width, ellipse_height), 
                0, 0, 360, 255, -1)
    
    # Apply mask
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    
    # Convert to HSV color space
    hsv_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
    
    # Filter out black pixels (masked out)
    non_zero_indices = np.where(mask > 0)
    non_zero_pixels = hsv_img[non_zero_indices]
    
    # Debug
    print(f"Analyzing {len(non_zero_pixels)} pixels after masking")
    
    # Skip processing if not enough pixels
    if len(non_zero_pixels) < 100:
        print("Warning: Not enough pixels for analysis")
        return "Unknown"
    
    # Reshape for k-means
    pixels = non_zero_pixels.reshape((-1, 3)).astype(np.float32)
    
    # Apply K-means clustering with more iterations for better convergence
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 15, cv2.KMEANS_RANDOM_CENTERS)
    
    # Find frequency of each cluster
    label_counts = Counter(labels.flatten())
    print(f"Cluster frequencies: {label_counts}")
    
    # Create detailed histograms for color analysis
    hue_hist = np.zeros(180)
    
    # Only consider pixels with reasonable saturation and value
    for pixel in non_zero_pixels:
        h, s, v = pixel
        if s > 20 and v > 40:
            hue_hist[int(h)] += 1
    
    # Smooth histogram
    hue_hist = np.convolve(hue_hist, np.ones(5)/5, mode='same')
    
    # Calculate range totals with CORRECTED blue/green ranges
    # Original blue range was too wide and overlapped with green
    green_range = np.sum(hue_hist[60:100])  # Green: 60-100
    blue_range = np.sum(hue_hist[100:140])  # Blue: 100-140
    red_range = np.sum(hue_hist[0:10]) + np.sum(hue_hist[170:180])
    yellow_range = np.sum(hue_hist[20:40])
    
    print(f"Hue distribution analysis (FIXED RANGES):")
    print(f"  Blue range (100-140): {blue_range}")
    print(f"  Green range (60-100): {green_range}")
    print(f"  Yellow range (20-40): {yellow_range}")
    print(f"  Red range (0-10, 170-180): {red_range}")
    
    # Get clusters with detailed color info
    cluster_data = []
    
    print("\nDetailed cluster analysis:")
    for i, center in enumerate(centers):
        h, s, v = center
        count = label_counts.get(i, 0)
        percentage = (count / len(labels)) * 100
        
        # Convert HSV center to BGR and RGB for comparison
        bgr = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_HSV2BGR)[0][0]
        rgb = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2RGB)[0][0]
        r, g, b = rgb
        
        print(f"Cluster {i}: HSV({h:.1f}, {s:.1f}, {v:.1f}), RGB({r},{g},{b}), Count: {count} ({percentage:.1f}%)")
        
        # Skip very dark or very light unsaturated colors
        if (v < 40) or (s < 25 and v > 200):
            print(f"  - REJECTED: Too dark/light or unsaturated")
            continue
        
        # RGB ratio check for blue/green distinction (critical for vehicles)
        # Blue should have b > g, while green has g > b
        color_classifier = "unknown"
        if b > g + 20:
            color_classifier = "likely blue (RGB ratio: b>g)"
        elif g > b + 20:
            color_classifier = "likely green (RGB ratio: g>b)"
        elif 100 <= h <= 140 and b >= g:
            color_classifier = "blue-ish (hue in blue range)"
        elif 60 <= h < 100 and g >= b:
            color_classifier = "green-ish (hue in green range)"
        
        # Calculate a weighted score based on count, saturation, and color clarity
        count_weight = count / len(labels)
        saturation_weight = s / 255.0
        value_weight = min(v / 255.0, 0.9)
        
        # RGB distinctiveness for blue/green
        rgb_distinctiveness = abs(b - g) / 255.0
        
        score = (count_weight * 0.5 + 
                saturation_weight * 0.25 + 
                value_weight * 0.1 + 
                rgb_distinctiveness * 0.15)
        
        cluster_data.append({
            'hsv': center,
            'rgb': rgb,
            'bgr': bgr,
            'count': count,
            'percentage': percentage,
            'score': score,
            'h': h,
            's': s,
            'v': v,
            'classifier': color_classifier
        })
        print(f"  - {color_classifier}")
        print(f"  - Score: {score:.4f}")
    
    # If no clusters passed filtering
    if not cluster_data:
        print("No suitable color clusters found")
        return "Unknown"
    
    # Sort clusters by score
    sorted_clusters = sorted(cluster_data, key=lambda x: x['score'], reverse=True)
    
    # Detailed analysis of top clusters
    print("\nTop clusters by score:")
    for i, cluster in enumerate(sorted_clusters[:3]):
        print(f"Top {i+1}: HSV({cluster['h']:.1f}, {cluster['s']:.1f}, {cluster['v']:.1f}), "
              f"RGB({cluster['rgb'][0]},{cluster['rgb'][1]},{cluster['rgb'][2]}), "
              f"Score: {cluster['score']:.4f}, {cluster['classifier']}")
    
    # Get the top scoring cluster
    top_cluster = sorted_clusters[0]
    h, s, v = top_cluster['hsv']
    r, g, b = top_cluster['rgb']
    
    # SPECIAL CASE: Blue Ford F-150 fix
    # For vehicles in the blue-green boundary, use both hue histogram and RGB channels
    if (90 <= h <= 110) or (abs(b - g) < 30):
        print("Vehicle in blue-green boundary zone, performing additional analysis...")
        
        # Trust the histogram distribution for overall color tendency
        if blue_range > green_range * 1.5:
            print("Histogram strongly suggests BLUE")
            if b >= g:  # Confirm with RGB check
                print("RGB values confirm blue (b>=g)")
                return "Blue"
        elif green_range > blue_range * 1.5:
            print("Histogram strongly suggests GREEN")
            if g >= b:  # Confirm with RGB check
                print("RGB values confirm green (g>=b)")
                return "Green"
        
        # If histogram is inconclusive, use RGB ratio as tiebreaker
        if b > g + 10:
            print("Using RGB ratio as tiebreaker: BLUE (b>g)")
            return "Blue"
        elif g > b + 10:
            print("Using RGB ratio as tiebreaker: GREEN (g>b)")
            return "Green"
        
        # Edge case: check the 2nd highest cluster if nearly tied
        if len(sorted_clusters) > 1:
            second_cluster = sorted_clusters[1]
            h2, s2, v2 = second_cluster['hsv']
            r2, g2, b2 = second_cluster['rgb']
            
            # If second cluster is more distinctly blue or green
            if b2 > g2 + 20:
                print("Second cluster is distinctly BLUE, using that")
                return "Blue"
            elif g2 > b2 + 20:
                print("Second cluster is distinctly GREEN, using that")
                return "Green"
    
    # Run the refined color classification for the final decision
    final_color = refined_vehicle_color_classification(h, s, v, r, g, b)
    print(f"Final color decision: {final_color}")
    return final_color

def refined_vehicle_color_classification(h, s, v, r, g, b):
    """Vehicle color classification using both HSV and RGB analysis"""
    print(f"Final classification - HSV({h:.1f}, {s:.1f}, {v:.1f}), RGB({r},{g},{b})")
    
    # Very low saturation = grayscale
    if s < 25:
        if v < 80:
            return "Black"
        elif v > 180:
            return "White"
        else:
            return "Gray/Silver"
    
    # FIXED: Blue-Green Classification Logic
    # For the blue-green boundary, use both hue and RGB channels
    
    # Case 1: Clear blue by hue and RGB ratio
    if 100 <= h <= 140 and b > g:
        # Stronger blue has higher difference between b and g
        if b > g + 20:
            return "Blue"
        else:
            # Metallic blues often have low saturation
            if s < 70:
                return "Light Blue"
            else:
                return "Blue"
    
    # Case 2: Clear green by hue and RGB ratio
    if 60 <= h < 100 and g > b:
        # Stronger green has higher difference
        if g > b + 20:
            return "Green" 
        else:
            # Olive greens have lower saturation
            if s < 70:
                return "Olive Green"
            else:
                return "Green"
    
    # Case 3: Boundary cases - use RGB as tiebreaker
    if 90 <= h <= 110:  # Boundary hue zone
        # Let RGB be the deciding factor
        if b > g:
            return "Blue"
        else:
            return "Green"
    
    # Yellow range
    if 20 <= h <= 40:
        if s > 100:
            return "Yellow"
        else:
            return "Gold/Beige"
            
    # Red range (handles wrap-around in HSV)
    if (h < 10 or h > 170) and s > 50:
        return "Red"
    
    # Orange is between red and yellow
    if 10 <= h < 20 and s > 50:
        return "Orange"
    
    # Teal/Cyan range - now narrowed
    if 80 <= h < 95 and b > r:
        return "Teal"
    
    # Purple range
    if 140 < h < 170:
        return "Purple"
    
    # Fallback based on RGB ratio for blue vs green when in doubt
    if b > g + 10:
        return "Blue"
    elif g > b + 10:
        return "Green"
    
    # Final fallback
    if s < 60 and v > 100:
        return "Metallic/Silver"
    
    return "Unknown"

# Enhanced visualization with RGB analysis
def visualize_color_analysis(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image for visualization.")
        return
    
    # Create mask focusing on vehicle body
    height, width = image.shape[:2]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    center_y = int(height * 0.5)
    center_x = width // 2
    ellipse_width = int(width * 0.5)
    ellipse_height = int(height * 0.35)
    cv2.ellipse(mask, (center_x, center_y), (ellipse_width, ellipse_height), 
                0, 0, 360, 255, -1)
    
    # Apply mask
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    
    # Convert to HSV and BGR
    hsv_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
    
    # Filter out zeros
    non_zero_pixels = hsv_img[mask > 0]
    non_zero_bgr = image[mask > 0]
    
    # Calculate RGB channel distributions
    b_vals = non_zero_bgr[:, 0]
    g_vals = non_zero_bgr[:, 1]
    r_vals = non_zero_bgr[:, 2]
    
    print(f"RGB Analysis:")
    print(f"  Mean B: {np.mean(b_vals):.1f}, Mean G: {np.mean(g_vals):.1f}, Mean R: {np.mean(r_vals):.1f}")
    print(f"  B>G pixels: {np.sum(b_vals > g_vals)}, G>B pixels: {np.sum(g_vals > b_vals)}")
    print(f"  B>G by 20+: {np.sum(b_vals > g_vals+20)}, G>B by 20+: {np.sum(g_vals > b_vals+20)}")
    
    # Extract hue values for histogram
    hues = []
    for pixel in non_zero_pixels:
        h, s, v = pixel
        if s > 20 and v > 40:
            hues.append(int(h))
    
    # Create hue histogram
    hue_hist, bin_edges = np.histogram(hues, bins=180, range=(0, 180))
    hue_hist_smoothed = np.convolve(hue_hist, np.ones(5)/5, mode='same')
    
    # Apply K-means
    k = 5
    pixels = non_zero_pixels.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 15, cv2.KMEANS_RANDOM_CENTERS)
    
    # Get cluster frequencies
    label_counts = Counter(labels.flatten())
    
    # Convert centers to BGR and RGB
    centers_info = []
    for center in centers:
        h, s, v = center
        bgr = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_HSV2BGR)[0][0]
        rgb = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2RGB)[0][0]
        r, g, b = rgb
        
        # Determine if likely blue or green based on RGB
        if b > g + 20:
            color_type = "BLUE"
        elif g > b + 20:
            color_type = "GREEN"
        elif 100 <= h <= 140 and b >= g:
            color_type = "blue-ish"
        elif 60 <= h < 100 and g >= b:
            color_type = "green-ish"
        else:
            color_type = "other"
            
        centers_info.append({
            'hsv': center,
            'bgr': bgr,
            'rgb': rgb,
            'color_type': color_type
        })
    
    # Save diagnostic images
    cv2.imwrite("./debug/debug_original.jpg", image)
    cv2.imwrite("./debug/debug_mask.jpg", mask)
    cv2.imwrite("./debug/debug_masked.jpg", masked_img)
    
    # Create color palette of dominant colors
    sorted_colors = sorted([(i, count) for i, count in label_counts.items()], 
                          key=lambda x: x[1], reverse=True)
    
    palette = np.zeros((100, k*100, 3), dtype=np.uint8)
    for i, (cluster_idx, count) in enumerate(sorted_colors):
        percentage = (count / len(labels)) * 100
        color = centers_info[cluster_idx]['bgr'].astype(int)
        color_type = centers_info[cluster_idx]['color_type']
        h, s, v = centers_info[cluster_idx]['hsv']
        r, g, b = centers_info[cluster_idx]['rgb']
        
        palette[:, i*100:(i+1)*100] = color
        print(f"Color {i+1}: HSV=({h:.1f}, {s:.1f}, {v:.1f}), RGB=({r},{g},{b}), "
              f"{percentage:.1f}%, {color_type}")
    
    cv2.imwrite("./debug/debug_palette.jpg", palette)
    
    # Create improved hue histogram visualization with corrected ranges
    hist_img_height = 200
    hist_img_width = 360
    hist_img = np.zeros((hist_img_height, hist_img_width, 3), dtype=np.uint8)
    
    # Normalize histogram for visualization
    if np.max(hue_hist_smoothed) > 0:
        normalized_hist = hue_hist_smoothed / np.max(hue_hist_smoothed) * (hist_img_height - 20)
        
        # Draw the histogram bars with their actual hue colors
        for x in range(180):
            # Create color with this hue, full saturation and value
            color_hsv = np.uint8([[[x, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            
            # Draw vertical line with height proportional to frequency
            y_height = int(normalized_hist[x])
            if y_height > 0:
                cv2.line(hist_img, (x*2, hist_img_height), (x*2, hist_img_height - y_height), 
                         color_bgr.tolist(), 2)
    
    # Mark CORRECTED color regions on histogram
    cv2.rectangle(hist_img, (20*2, 10), (40*2, 30), (0, 255, 255), 2)  # Yellow
    cv2.putText(hist_img, "Yellow", (20*2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.rectangle(hist_img, (60*2, 10), (100*2, 30), (0, 255, 0), 2)  # Green (FIXED)
    cv2.putText(hist_img, "Green", (70*2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.rectangle(hist_img, (100*2, 10), (140*2, 30), (255, 0, 0), 2)  # Blue (FIXED)
    cv2.putText(hist_img, "Blue", (110*2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imwrite("./debug/debug_hue_histogram.jpg", hist_img)
    
    # Create RGB comparison visualization
    rgb_img_height = 200
    rgb_img_width = 400
    rgb_img = np.zeros((rgb_img_height, rgb_img_width, 3), dtype=np.uint8)
    
    # Draw columns for top clusters showing RGB values
    for i, (cluster_idx, count) in enumerate(sorted_colors[:min(3, len(sorted_colors))]):
        if i >= 3:
            break
            
        rgb = centers_info[cluster_idx]['rgb']
        r, g, b = map(int, rgb)
        
        # Draw three bars showing R, G, B channel values
        x_start = i * 130 + 20
        
        # Red bar
        cv2.rectangle(rgb_img, (x_start, rgb_img_height-20), 
                     (x_start+30, rgb_img_height-20-r), (0, 0, 255), -1)
        cv2.putText(rgb_img, f"R:{r}", (x_start, rgb_img_height-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Green bar
        cv2.rectangle(rgb_img, (x_start+40, rgb_img_height-20), 
                     (x_start+70, rgb_img_height-20-g), (0, 255, 0), -1)
        cv2.putText(rgb_img, f"G:{g}", (x_start+40, rgb_img_height-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Blue bar
        cv2.rectangle(rgb_img, (x_start+80, rgb_img_height-20), 
                     (x_start+110, rgb_img_height-20-b), (255, 0, 0), -1)
        cv2.putText(rgb_img, f"B:{b}", (x_start+80, rgb_img_height-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Label for cluster
        cv2.putText(rgb_img, f"Cluster {i+1}", (x_start+30, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imwrite("./debug/debug_rgb_comparison.jpg", rgb_img)
    
    print("Debug images saved: debug_original.jpg, debug_mask.jpg, debug_masked.jpg, "
          "debug_palette.jpg, debug_hue_histogram.jpg, debug_rgb_comparison.jpg")

# # Call visualization before main color detection
# visualize_color_analysis(truck_image_path)
# Open video capture
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open camera stream.")
else:
    ret, frame = cap.read()
    if ret:
        # Save the screenshot
        screenshot_path = "screenshot.jpg"
        cv2.imwrite(screenshot_path, frame)
        print(f"Screenshot saved as {screenshot_path}")

        # Call this right before your main color detection
        visualize_color_analysis(truck_image_path)

        # Detect truck color
        truck_color = get_dominant_color(truck_image_path)

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
        json_output_path = "output.txt"
        with open(json_output_path, "w") as json_file:
            json.dump(test_data, json_file, indent=2)
        print(f"Output written to {json_output_path}")

    else:
        print("Error: Could not capture frame.")

# Release the capture
cap.release()


# import cv2
# import json
# import random
# import datetime

# # RTSP stream URL
# rtsp_url = "rtsp://admin:admin123@192.168.1.16:554/cam/realmonitor?channel=1&subtype=0"

# # Open video capture
# cap = cv2.VideoCapture(rtsp_url)

# if not cap.isOpened():
#     print("Error: Could not open camera stream.")
# else:
#     ret, frame = cap.read()
#     if ret:
#         # Save the screenshot
#         screenshot_path = "screenshot.jpg"
#         cv2.imwrite(screenshot_path, frame)
#         print(f"Screenshot saved as {screenshot_path}")

#         # Generate test JSON data
#         timestamp = datetime.datetime.now().isoformat()
#         test_data = [
#             {
#                 "metadata": {
#                     "location_id": "Test_Location",
#                     "box_id": "Test_Camera",
#                     "datetime": timestamp
#                 },
#                 "vehicle_data": {
#                     "plate_no": f"XYZ{random.randint(100, 999)}",
#                     "usdot": str(random.randint(1000000, 9999999)),
#                     "vin": f"VIN{random.randint(100000, 999999)}",
#                     "truck_color": random.choice(["Red", "Blue", "Green", "Black"]),
#                     "trailers": [f"TRAILER{random.randint(100, 999)}" for _ in range(random.randint(0, 2))]
#                 },
#                 "model_version": "1.0"
#             }
#         ]

#         # Save JSON output to a .txt file
#         json_output_path = "output.txt"
#         with open(json_output_path, "w") as json_file:
#             json.dump(test_data, json_file, indent=2)
#         print(f"Output written to {json_output_path}")

#     else:
#         print("Error: Could not capture frame.")

# # Release the capture
# cap.release()
