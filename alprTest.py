from openalpr import Alpr

# Path to your test image
image_path = "./testimgs/alpr_test1.jpg"

# Initialize OpenALPR (Change "us" to your country code if needed)
alpr = Alpr("us", "/etc/openalpr/openalpr.conf", "/usr/share/openalpr/runtime_data")


'''
{'version': 2, 'data_type': 'alpr_results', 'epoch_time': 1739933545086, 'img_width': 1920, 'img_height': 1080, 'processing_time_ms': 1504.720093, 'regions_of_interest': [{'x': 0, 'y': 0, 'width': 1920, 'height': 1080}], 'results': [{'plate': 'PA47773', 'confidence': 89.916176, 'matches_template': 0, 'plate_index': 0, 'region': '', 'region_confidence': 0, 'processing_time_ms': 52.115311, 'requested_topn': 25, 'coordinates': [{'x': 1114, 'y': 806}, {'x': 1207, 'y': 796}, {'x': 1212, 'y': 861}, {'x': 1118, 'y': 871}], 'candidates': [{'plate': 'PA47773', 'confidence': 89.916176, 'matches_template': 0}, {'plate': 'PAA7773', 'confidence': 81.712158, 'matches_template': 0}, {'plate': 'PA7773', 'confidence': 81.426033, 'matches_template': 0}]}, {'plate': 'BD1I', 'confidence': 77.10321, 'matches_template': 0, 'plate_index': 1, 'region': '', 'region_confidence': 0, 'processing_time_ms': 268.43927, 'requested_topn': 25, 'coordinates': [{'x': 918, 'y': 449}, {'x': 1250, 'y': 408}, {'x': 1265, 'y': 539}, {'x': 933, 'y': 580}], 'candidates': [{'plate': 'BD1I', 'confidence': 77.10321, 'matches_template': 0}, {'plate': 'DD1I', 'confidence': 77.07682, 'matches_template': 0}, {'plate': 'BD11', 'confidence': 76.299103, 'matches_template': 0}, {'plate': 'DD11', 'confidence': 76.272713, 'matches_template': 0}, {'plate': 'OD1I', 'confidence': 74.638718, 'matches_template': 0}, {'plate': 'BQ1I', 'confidence': 74.53923, 'matches_template': 0}, {'plate': 'DQ1I', 'confidence': 74.51284, 'matches_template': 0}, {'plate': 'JD1I', 'confidence': 74.17067, 'matches_template': 0}, {'plate': '3D1I', 'confidence': 73.865288, 'matches_template': 0}, {'plate': 'OD11', 'confidence': 73.83461, 'matches_template': 0}, {'plate': '8D1I', 'confidence': 73.763504, 'matches_template': 0}, {'plate': '9D1I', 'confidence': 73.757317, 'matches_template': 0}, {'plate': 'BQ11', 'confidence': 73.73513, 'matches_template': 0}, {'plate': 'ZD1I', 'confidence': 73.728935, 'matches_template': 0}, {'plate': 'QD1I', 'confidence': 73.710518, 'matches_template': 0}, {'plate': 'DQ11', 'confidence': 73.70874, 'matches_template': 0}, {'plate': '0D1I', 'confidence': 73.614265, 'matches_template': 0}, {'plate': 'WD1I', 'confidence': 73.503487, 'matches_template': 0}, {'plate': 'UD1I', 'confidence': 73.435532, 'matches_template': 0}, {'plate': 'JD11', 'confidence': 73.366562, 'matches_template': 0}, {'plate': 'RD1I', 'confidence': 73.340065, 'matches_template': 0}, {'plate': 'HD1I', 'confidence': 73.286919, 'matches_template': 0}, {'plate': 'GD1I', 'confidence': 73.081367, 'matches_template': 0}, {'plate': '3D11', 'confidence': 73.06118, 'matches_template': 0}, {'plate': '8D11', 'confidence': 72.959396, 'matches_template': 0}]}]}
'''

if not alpr.is_loaded():
    print("Error: OpenALPR failed to load")
    exit()

# Run ALPR on the image
results = alpr.recognize_file(image_path)

# Print results for top 2 plates
for result in results['results']:
    print(f"Plate: {result['plate']} | Confidence: {result['confidence']:.2f}")

# Clean up
alpr.unload()
