import os
import glob
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer

# Paths
input_folder = 'downsampled_images_32'
output_folder = 'results/gfpgan_32'
os.makedirs(output_folder, exist_ok=True)

# Set GFPGAN parameters
model_path = 'D:\cv_project\GFPGAN\experiments\pretrained_models\GFPGANv1.3.pth'     # GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth
upscale = 32
arch = 'clean'  # or 'original' if needed
channel_multiplier = 2

# Initialize GFPGAN
restorer = GFPGANer(
    model_path=model_path,
    upscale=upscale,
    arch=arch,
    channel_multiplier=channel_multiplier,
    bg_upsampler=None
)

# Process images
image_paths = glob.glob(os.path.join(input_folder, '*'))

for img_path in image_paths:
    img_name = os.path.basename(img_path)
    print(f'Processing: {img_name}')
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Enhance
    cropped_faces, restored_faces, restored_img = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)

    # Save result
    save_path = os.path.join(output_folder, f'enhanced_{img_name}')
    cv2.imwrite(save_path, restored_img)
    print(f'Saved to: {save_path}')

