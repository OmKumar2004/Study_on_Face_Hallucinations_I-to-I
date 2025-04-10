import os
import lpips
import torch
from PIL import Image
from torchvision import transforms

# Initialize LPIPS model
loss_fn = lpips.LPIPS(net='alex')  # or 'vgg'

# Directories
original_dir = 'original_images'
model_variants = {
    'codeformer': ['codeformer_32', 'codeformer_64', 'codeformer_128', 'codeformer_256'],
    'gpen': ['gpen_32', 'gpen_64', 'gpen_128', 'gpen_256'],
    'gfpgan': ['gfpgan']  # Only one variant
}

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# LPIPS computation
def compute_lpips(img1_path, img2_path):
    img1 = transform(Image.open(img1_path).convert('RGB')).unsqueeze(0)
    img2 = transform(Image.open(img2_path).convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        distance = loss_fn(img1, img2)
    return distance.item()

# Custom filename pattern logic
def get_model_filename(model, orig_filename):
    base = os.path.splitext(orig_filename)[0]
    if model == 'codeformer':
        return f"{base}_restored.png"
    elif model == 'gpen':
        return f"{base}_gpen_1024.png"
    elif model == 'gfpgan':
        return f"enhanced_{base}.png"
    return None

# Open log file
log_file_path = 'lpips_log.txt'
with open(log_file_path, 'w') as log_file:
    for model_name, variants in model_variants.items():
        for variant in variants:
            model_dir = os.path.join('results', variant)
            log_file.write(f"\n=== LPIPS Scores for {variant.upper()} ===\n")
            total_score = 0
            count = 0
            for filename in os.listdir(original_dir):
                orig_path = os.path.join(original_dir, filename)
                model_filename = get_model_filename(model_name, filename)
                upsampled_path = os.path.join(model_dir, model_filename)
                if os.path.exists(upsampled_path):
                    score = compute_lpips(orig_path, upsampled_path)
                    log_file.write(f"{filename} vs {model_filename}: {score:.4f}\n")
                    total_score += score
                    count += 1
                else:
                    log_file.write(f"Warning: {model_filename} not found in {model_dir}\n")
            if count > 0:
                avg_score = total_score / count
                log_file.write(f"Average LPIPS for {variant.upper()}: {avg_score:.4f}\n")

print(f"\nLPIPS scores have been saved to '{log_file_path}'")
