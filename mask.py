import cv2
import numpy as np

# Load image
image = cv2.imread("image.jpg")

# Get image dimensions
height, width = image.shape[:2]
center_x, center_y = width // 2, height // 2

# Create distance map
Y, X = np.ogrid[:height, :width]
distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

# Normalize distances (0 at center, 1 at radius limit)
radius = min(width, height) // 3  # Radius defining importance region
normalized_distance = np.clip(distance / radius, 0, 1)  # Ensures values are between 0-1

# Apply EaseInQuart function (t^4)
mask = 1 - normalized_distance**2  # Inverts so center is high importance

# Convert to 8-bit grayscale (0-255)
mask = (mask * 255).astype(np.uint8)

# Compute histogram using the custom dropoff mask
histogram = cv2.calcHist([image], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])

# Normalize histogram
histogram /= np.sum(histogram)

# Show the mask for debugging
cv2.imshow("EaseInQuart Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
