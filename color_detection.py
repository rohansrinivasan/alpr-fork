import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import time
import sys
def downsample_image(image, max_dim=256):
    old_size = image.shape[:2]
    scale = max_dim / max(old_size)
    image = cv2.resize(image, (int(old_size[1] * scale), int(old_size[0] * scale)))
    return image

def get_dominant_color(image_path, n_colors=3):
    # Read the image
    image = cv2.imread(image_path)
    # downsample to image to have max dimension of 256
    image = downsample_image(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Get the most dominant color
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Count labels to find the most dominant cluster
    counter = Counter(labels)
    dominant_color = colors[counter.most_common(1)[0][0]]
    
    return map_to_basic_color(dominant_color)

def map_to_basic_color(rgb_color):
    # Define color ranges (in RGB) with expanded ranges for priority colors
    color_ranges = {
        'White': ([180, 180, 180], [255, 255, 255]),    # Expanded white range
        'Metallic': ([90, 90, 90], [180, 180, 180]),    # Expanded metallic range
        'Blue': ([0, 0, 130], [80, 80, 255]),           # Expanded blue range
        'Black': ([0, 0, 0], [60, 60, 60]),             # Expanded black range
        'Green': ([0, 130, 0], [70, 255, 70]),          # Adjusted green range
        'Red': ([130, 0, 0], [255, 70, 70]),            # Adjusted red range
        'Yellow': ([200, 200, 0], [255, 255, 60]),      # Original yellow
        'Orange': ([200, 100, 0], [255, 160, 60])       # Original orange
    }
    
    r, g, b = rgb_color
    
    # Check if the color falls within any of the defined ranges
    for color_name, ((r1, g1, b1), (r2, g2, b2)) in color_ranges.items():
        if (r1 <= r <= r2 and 
            g1 <= g <= g2 and 
            b1 <= b <= b2):
            return color_name
    
    # If no match is found, find the closest basic color
    min_distance = float('inf')
    closest_color = None
    
    for color_name, ((r1, g1, b1), _) in color_ranges.items():
        distance = np.sqrt((r - r1)**2 + (g - g1)**2 + (b - b1)**2)
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
    
    return closest_color

# Example usage
if __name__ == "__main__":
    # get image from command line
    image_path = "testimgs/alpr_test1.jpg"
    if len(sys.argv) == 2:
        image_path = sys.argv[1]
    else:
        print(f"No image path provided, using default image: {image_path}")
    t1 = time.time()
    dominant_color = get_dominant_color(image_path)
    t2 = time.time()
    print(f"The dominant color is: {dominant_color}")
    print(f"Time taken: {t2 - t1} seconds")