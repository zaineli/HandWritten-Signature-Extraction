import os
import numpy as np
from PIL import Image
from math import atan2, degrees

# Helper function: Find the centroid of a given cell
def find_centroid(image, left, right, top, bottom):
    cx, cy, n = 0, 0, 0
    for x in range(left, right):
        for y in range(top, bottom):
            if image.getpixel((x, y)) == 0:  # Black pixel
                cx += x
                cy += y
                n += 1
    return (cx / n, cy / n) if n != 0 else (0, 0)

# Helper function: Calculate black-to-white transitions in a given cell
def count_transitions(image, left, right, top, bottom):
    transitions = 0
    prev = image.getpixel((left, top))
    for x in range(left, right):
        for y in range(top, bottom):
            curr = image.getpixel((x, y))
            if curr == 255 and prev == 0:  # Black-to-white transition
                transitions += 1
            prev = curr
    return transitions

# Helper function: Calculate aspect ratio of a given cell
def find_aspect_ratio(left, right, top, bottom):
    return (right - left) / (bottom - top)

# Helper function: Calculate skew angle using horizontal projection profile
def calculate_skew_angle(image, left, right, top, bottom):
    horizontal_profile = np.sum(np.array(image.crop((left, top, right, bottom))), axis=1)
    max_profile = np.argmax(horizontal_profile)
    skew_angle = atan2(max_profile - len(horizontal_profile) // 2, right - left)
    return degrees(skew_angle)

# Helper function: Calculate slant angle using vertical projection profile
def calculate_slant_angle(image, left, right, top, bottom):
    vertical_profile = np.sum(np.array(image.crop((left, top, right, bottom))), axis=0)
    max_profile = np.argmax(vertical_profile)
    slant_angle = atan2(max_profile - len(vertical_profile) // 2, bottom - top)
    return degrees(slant_angle)

# Recursive function to split the image and extract features
def recursive_split(image, left, right, top, bottom, depth=0, centroids=[], transitions=[], ratios=[], skews=[], slants=[]):
    if depth == 3:  # Stop at depth 3 (64 cells)
        cx, cy = find_centroid(image, left, right, top, bottom)
        trans = count_transitions(image, left, right, top, bottom)
        ratio = find_aspect_ratio(left, right, top, bottom)
        skew = calculate_skew_angle(image, left, right, top, bottom)
        slant = calculate_slant_angle(image, left, right, top, bottom)

        # Collect the data for later saving
        centroids.append((cx, cy))
        transitions.append(trans)
        ratios.append(ratio)
        skews.append(skew)
        slants.append(slant)
        return

    mid_x = (left + right) // 2
    mid_y = (top + bottom) // 2

    # Recursively divide into four segments
    recursive_split(image, left, mid_x, top, mid_y, depth + 1, centroids, transitions, ratios, skews, slants)  # Top-left
    recursive_split(image, mid_x, right, top, mid_y, depth + 1, centroids, transitions, ratios, skews, slants)  # Top-right
    recursive_split(image, left, mid_x, mid_y, bottom, depth + 1, centroids, transitions, ratios, skews, slants)  # Bottom-left
    recursive_split(image, mid_x, right, mid_y, bottom, depth + 1, centroids, transitions, ratios, skews, slants)  # Bottom-right

# Process multiple reference images
def process_images():
    os.makedirs('processed/centroid', exist_ok=True)
    os.makedirs('processed/ratio', exist_ok=True)
    os.makedirs('processed/transitions', exist_ok=True)
    os.makedirs('processed/skew', exist_ok=True)
    os.makedirs('processed/slant', exist_ok=True)

    feature_data = []  # Store all transitions for comparison

    for i in range(1, 26):  # Process R001.png to R025.png
        filename = f'./Reference/R{i:03}.png'
        if not os.path.exists(filename):
            print(f"File {filename} not found. Skipping...")
            continue

        print(f"Processing {filename}...")
        img = Image.open(filename).convert('L')  # Convert to grayscale
        bw_img = img.point(lambda x: 0 if x < 128 else 255, '1')  # Binarize

        width, height = bw_img.size
        left, right, top, bottom = width, 0, height, 0

        # Find bounding box
        for x in range(width):
            for y in range(height):
                if bw_img.getpixel((x, y)) == 0:
                    left = min(left, x)
                    right = max(right, x)
                    top = min(top, y)
                    bottom = max(bottom, y)

        centroids, transitions, ratios, skews, slants = [], [], [], [], []
        recursive_split(bw_img, left, right, top, bottom, centroids=centroids, transitions=transitions, ratios=ratios, skews=skews, slants=slants)

        # Save feature data
        img_id = f'R{i:03}'
        np.savetxt(f'processed/centroid/{img_id}.txt', centroids, fmt='%.2f', header='cx cy')
        np.savetxt(f'processed/transitions/{img_id}.txt', transitions, fmt='%d', header='transitions')
        np.savetxt(f'processed/ratio/{img_id}.txt', ratios, fmt='%.2f', header='aspect_ratios')
        np.savetxt(f'processed/skew/{img_id}.txt', skews, fmt='%.2f', header='skew_angles')
        np.savetxt(f'processed/slant/{img_id}.txt', slants, fmt='%.2f', header='slant_angles')

        feature_data.append(transitions)  # Collect transitions for comparison

    print("Feature extraction and saving complete!")
    compare_features(feature_data)

# Compare black-to-white transitions across all images
def compare_features(feature_data):
    print("Comparing features across all signatures...")

    stable_cells = np.zeros(64)  # Store stability (1 = stable, 0 = unstable)

    for i in range(len(feature_data) - 1):
        for j in range(64):  # Compare each of the 64 cells
            if feature_data[i][j] == feature_data[i + 1][j]:  # Stable if transitions match
                stable_cells[j] += 1

    # Analyze stability
    stable_threshold = len(feature_data) - 1  # Maximum stability
    stable_cells = [1 if count == stable_threshold else 0 for count in stable_cells]

    # show stable cells as numpy array
    print(f"Stable cells: {sum(stable_cells)}/64")

    # Save stability results
    np.savetxt('processed/stable_cells.txt', stable_cells, fmt='%d', header='Stable cells (1=stable, 0=unstable)')

    print("Feature comparison complete!")

# Main function to run the entire process
if __name__ == "__main__":
    process_images()
