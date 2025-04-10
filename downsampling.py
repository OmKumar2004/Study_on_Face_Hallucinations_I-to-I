import os
import numpy as np
import cv2
from PIL import Image

# Parameters
SCALE_FACTOR = 4         # e.g. downsample by 2
BLUR_KERNEL = 5          # must be odd, e.g. 3, 5, 7
NOISE_SIGMA = 5          # standard deviation of Gaussian noise
JPEG_QUALITY = 70        # JPEG compression quality

INPUT_DIR = 'original_images'
OUTPUT_DIR = 'downsampled_images_256'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Degradation function
def degrade_image(img):
    # Convert to numpy array if it's a PIL image
    if isinstance(img, Image.Image):
        img = np.array(img)

    # 1. Gaussian blur
    blurred = cv2.GaussianBlur(img, (BLUR_KERNEL, BLUR_KERNEL), 0)

    # 2. Downsample
    h, w = img.shape[:2]
    lr = cv2.resize(blurred, (w // SCALE_FACTOR, h // SCALE_FACTOR), interpolation=cv2.INTER_AREA)

    # 3. Add Gaussian noise
    noise = np.random.normal(0, NOISE_SIGMA, lr.shape).astype(np.int16)
    noisy = np.clip(lr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 4. JPEG compression
    _, encimg = cv2.imencode('.jpg', noisy, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    degraded = cv2.imdecode(encimg, 1)

    return degraded

# Process all images
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        input_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(input_path)

        if img is None:
            print(f"Could not read {filename}")
            continue

        degraded_img = degrade_image(img)

        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, degraded_img)
        print(f"Saved degraded image: {output_path}")

































# import os
# import cv2

# # Parameters
# SCALE_FACTOR = 2  # Change this according to how much you want to downsample
# INPUT_DIR = 'original_images'
# OUTPUT_DIR = 'downsampled_images'

# # Create output directory if it doesn't exist
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Loop through images
# for filename in os.listdir(INPUT_DIR):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#         img_path = os.path.join(INPUT_DIR, filename)
#         img = cv2.imread(img_path)

#         if img is None:
#             print(f"Failed to load image: {filename}")
#             continue

#         # Downsample
#         h, w = img.shape[:2]
#         downsampled = cv2.resize(img, (w // SCALE_FACTOR, h // SCALE_FACTOR), interpolation=cv2.INTER_AREA)

#         # Save
#         output_path = os.path.join(OUTPUT_DIR, filename)
#         cv2.imwrite(output_path, downsampled)

#         print(f"Saved downsampled image: {output_path}")
