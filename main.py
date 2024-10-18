from PIL import Image
import numpy as np

img = Image.open('R001.png')


# Step 1: Bounding Box Detection

# Convert the image to grayscale
# The 'L' mode means that each pixel in the image is
# represented by a single value (from 0 to 255)
img = img.convert('L')

# Convert the image to a numpy array
# '1' mode: This converts the image to binary (black and white) format
bw_img = img.point(lambda x: 0 if x < 128 else 255, '1')


width, height = bw_img.size

left, right, top, bottom = width, 0, height, 0

# Find bounding box coordinates
for x in range(width):
    for y in range(height):
        if bw_img.getpixel((x, y)) == 0:  # Black pixel
            left = min(left, x)
            right = max(right, x)
            top = min(top, y)
            bottom = max(bottom, y)

print(f"Bounding Box: Left={left}, Right={right}, Top={top}, Bottom={bottom}")



# Step 2: Centroid Calculation
def find_centroid(image, left, right, top, bottom):
    cx, cy, n = 0, 0, 0

    for x in range(left, right):
        for y in range(top, bottom):
            if image.getpixel((x, y)) == 0:  # Black pixel
                cx += x
                cy += y
                n += 1

    return (cx / n, cy / n) if n != 0 else (0, 0)

# Calculate centroid for the full signature
centroid = find_centroid(bw_img, left, right, top, bottom)
print(f"Centroid: {centroid}")


# Step 3: segment the signature into 4 equal parts from the centroid

def segment_signature(image, centroid, left, right, top, bottom):
    cx, cy = centroid

    # Create a new image with the same size as the original image
    segmented_img = Image.new('1', image.size, 1)

    # Segment the image into 4 equal parts
    for x in range(left, right):
        for y in range(top, bottom):
            if image.getpixel((x, y)) == 0:  # Black pixel
                if x < cx and y < cy:
                    segmented_img.putpixel((x, y), 0)
                elif x >= cx and y < cy:
                    segmented_img.putpixel((x, y), 2)
                elif x < cx and y >= cy:
                    segmented_img.putpixel((x, y), 3)
                else:
                    segmented_img.putpixel((x, y), 4)

    return segmented_img

segmented_img = segment_signature(bw_img, centroid, left, right, top, bottom)
segmented_img.show()
