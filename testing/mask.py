import cv2
import numpy as np

# Load image
image = cv2.imread("image.jpg")

# Get image dimensions
height, width = image.shape[:2]
center_x, center_y = width // 2, height // 2

# Create coordinate grid
Y, X = np.ogrid[:height, :width]

# Compute separate normalized distances for x and y (ellipse instead of circle)
normalized_x = ((X - center_x) / (width / 2)) ** 2
normalized_y = ((Y - center_y) / (height / 2)) ** 2

# Compute elliptical distance and apply quadratic dropoff
normalized_distance = np.sqrt(normalized_x + normalized_y)  # Ellipse-based distance
mask = np.clip(1 - normalized_distance ** 2, 0, 1)  # Quadratic dropoff

# Convert to 8-bit grayscale (0-255)
mask = (mask * 255).astype(np.uint8)

# Compute histogram using the oval mask
histogram = cv2.calcHist([image], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])

# Normalize histogram
histogram /= np.sum(histogram)

# Show the mask for debugging
cv2.imshow("Oval Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
