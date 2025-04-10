import os
import sys
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from basicsr.archs.codeformer_arch import CodeFormer

# Setup paths
codeformer_path = os.path.join(os.path.dirname(__file__), 'CodeFormer')
if codeformer_path not in sys.path:
    sys.path.insert(0, codeformer_path)

input_dir = 'downsampled_images_256'
output_dir = 'results/codeformer_256'
os.makedirs(output_dir, exist_ok=True)

# Load CodeFormer
net = CodeFormer(
    dim_embd=512,
    codebook_size=1024,
    n_head=8,
    n_layers=9,
    connect_list=['32', '64', '128', '256']
)
ckpt_path = 'CodeFormer/weights/CodeFormer.pth'
net.load_state_dict(torch.load(ckpt_path, map_location='cpu')['params_ema'], strict=True)
net.eval()
net.to('cpu')

# Inference loop (no face detection)
for img_name in os.listdir(input_dir):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        continue

    print(f"Processing {img_name}")
    img_path = os.path.join(input_dir, img_name)

    # Load and resize to 512x512
    img = Image.open(img_path).convert('RGB')
    img = img.resize((512, 512))  # Force resize

    # Convert to tensor
    img_tensor = to_tensor(img).unsqueeze(0).to('cpu')

    with torch.no_grad():
        output = net(img_tensor, w=0.7, adain=True)[0]
        restored_face = output.clamp(0, 1).cpu()
        restored_face = to_pil_image(restored_face.squeeze(0)).resize((1024, 1024), Image.BICUBIC)

    save_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_restored.png")
    restored_face.save(save_path)





# import sys
# import os

# # Set CodeFormer as root for all relative imports
# codeformer_path = os.path.join(os.path.dirname(__file__), 'CodeFormer')
# if codeformer_path not in sys.path:
#     sys.path.insert(0, codeformer_path)


# import os
# import torch
# from torchvision.transforms.functional import to_tensor, to_pil_image
# from PIL import Image
# from basicsr.archs.codeformer_arch import CodeFormer
# from facelib.utils.face_restoration_helper import FaceRestoreHelper
# from facelib.detection.retinaface.retinaface import RetinaFace

# # Paths
# input_dir = 'downsampled_images'
# output_dir = 'results/codeformer'
# os.makedirs(output_dir, exist_ok=True)

# # Load CodeFormer model
# net = CodeFormer(
#     dim_embd=512,
#     codebook_size=1024,
#     n_head=8,
#     n_layers=9,
#     connect_list=['32', '64', '128', '256']
# )
# ckpt_path = 'CodeFormer/weights/CodeFormer.pth'
# net.load_state_dict(torch.load(ckpt_path)['params_ema'], strict=True)
# net.eval()
# net.to('cpu')

# # Set up face helper with 4x upscaling
# face_helper = FaceRestoreHelper(
#     upscale_factor=4,
#     face_size=512,
#     crop_ratio=(1, 1),
#     det_model='retinaface_resnet50',
#     save_ext='png',
#     use_parse=True
# )
# face_helper.face_detector = RetinaFace()

# # Inference loop
# for img_name in os.listdir(input_dir):
#     if not img_name.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
#         continue
#     print(f"Processing {img_name}")
#     img_path = os.path.join(input_dir, img_name)

#     # Load and preprocess image
#     face_helper.clean_all()
#     face_helper.read_image(img_path)
#     face_helper.get_face_landmarks_5()
#     face_helper.align_warp_face()

#     if not face_helper.cropped_faces:
#         print(f"  No faces detected in {img_name}")
#         continue

#     # Process each face
#     for idx, cropped_face in enumerate(face_helper.cropped_faces):
#         face_tensor = to_tensor(cropped_face).unsqueeze(0).to('cpu')

#         with torch.no_grad():
#             output = net(face_tensor, w=0.7, adain=True)[0]
#             restored_face = output.clamp(0, 1).cpu()
#             restored_face = to_pil_image(restored_face.squeeze(0))

#         face_helper.add_restored_face(restored_face)

#     # Paste restored faces back into original image
#     final_image = face_helper.get_final_image(upscale=True)
#     save_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_restored.png")
#     final_image.save(save_path)

# print("All images enhanced and saved to results/codeformer/")
