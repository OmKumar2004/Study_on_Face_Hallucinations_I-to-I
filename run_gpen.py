import os
import sys
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np

# === Add GPEN path to sys ===
gpen_path = os.path.join(os.path.dirname(__file__), 'GPEN')
if gpen_path not in sys.path:
    sys.path.insert(0, gpen_path)

# === Import FaceGAN ===
from face_model.face_gan import FaceGAN

# === Input/output folders ===
input_folder = 'downsampled_images_256'
output_folder = 'results/gpen_256'
os.makedirs(output_folder, exist_ok=True)

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load GPEN model ===
model = FaceGAN(
    base_dir='gpen',                   # Folder containing weights/GPEN-BFR-512.pth
    model='GPEN-BFR-512',
    in_size=512,
    out_size=512,
    channel_multiplier=2,
    narrow=1,
    device=device
)

# === Image transforms ===
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# === Process all images ===
for img_name in os.listdir(input_folder):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue

    print(f"🔧 Processing {img_name}")
    img_path = os.path.join(input_folder, img_name)
    img = Image.open(img_path).convert("RGB")

    # Upscale to 512x512
    img_resized = img.resize((512, 512), resample=Image.BICUBIC)
    img_np = np.array(img_resized)[:, :, ::-1]  # Convert to BGR for GPEN

    # GPEN enhancement
    output_np = model.process(img_np)

    # Convert to PIL and upscale to 1024x1024
    output_img = Image.fromarray(output_np[:, :, ::-1])  # BGR to RGB
    output_img_1024 = output_img.resize((1024, 1024), resample=Image.BICUBIC)

    # Save result
    out_path = os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}_gpen_1024.png")
    output_img_1024.save(out_path)

