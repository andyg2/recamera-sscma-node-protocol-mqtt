import cv2
import numpy as np
import os

# The creates an oval mast to fill the bounding box portion of the image. It is a quadratic mask (similar to gaussian blur but much more efficient).
# The mask is used to compute the histogram of the image, giving more weight to the center of the image.

factor = 1.3

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the image
image_path = os.path.join(script_dir, "image.jpg")

# Load image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Unable to load image at path '{image_path}'. Please check the file path.")
    exit()

# Get image dimensions
height, width = image.shape[:2]
center_x, center_y = width // 2, height // 2

# Create coordinate grid
Y, X = np.ogrid[:height, :width]

# Compute separate normalized distances for x and y (ellipse instead of circle)
normalized_x = ((X - center_x) / (width / 2)) ** 2
normalized_y = ((Y - center_y) / (height / 2)) ** 2

# Compute elliptical distance and apply quadratic dropoff
normalized_distance = np.sqrt(normalized_x + normalized_y)
mask = np.clip(1 - normalized_distance ** factor, 0, 1)

# Convert mask to 3-channel float32
mask = mask.astype(np.float32)
mask = cv2.merge([mask, mask, mask])

# Apply the mask using multiplication instead of bitwise operations
final_image = (image.astype(np.float32) * mask).astype(np.uint8)

# Display the final image
cv2.imshow("Original Image with Gradual Fade", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
