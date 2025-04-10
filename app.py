# app.py
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import umap
from io import BytesIO
from PIL import Image


def load_image(uploaded_file, resize_dim=(300, 300)):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(resize_dim)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def analyze_images(img1, img2, title):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # Draw matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    st.image(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB),
             caption=f"SIFT Matches - {title} | Good Matches: {len(good_matches)}",
             use_container_width=True)

    # Keypoint Size Distribution
    sizes1 = [kp.size for kp in kp1]
    sizes2 = [kp.size for kp in kp2]
    fig, ax = plt.subplots()
    ax.hist(sizes1, bins=30, alpha=0.5, label='Image 1')
    ax.hist(sizes2, bins=30, alpha=0.5, label='Image 2')
    ax.set_title('Keypoint Size Distribution')
    ax.set_xlabel('Size')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

    # Orientation
    angles1 = [kp.angle for kp in kp1]
    angles2 = [kp.angle for kp in kp2]
    fig, ax = plt.subplots()
    ax.hist(angles1, bins=36, alpha=0.5, label='Image 1')
    ax.hist(angles2, bins=36, alpha=0.5, label='Image 2')
    ax.set_title('Keypoint Orientation Distribution')
    ax.set_xlabel('Angle')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

    # UMAP visualization
    if des1 is not None:
        sample_des = des1 if des1.shape[0] < 100 else des1[np.random.choice(des1.shape[0], 100, replace=False)]
        reducer = umap.UMAP(random_state=42)
        des_umap = reducer.fit_transform(sample_des)
        fig, ax = plt.subplots()
        ax.scatter(des_umap[:, 0], des_umap[:, 1], s=10, c='purple')
        ax.set_title("UMAP of SIFT Descriptors (Image 1)")
        st.pyplot(fig)

    # Heatmap
    heatmap = np.zeros(img1_gray.shape)
    for kp in kp1:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= x < heatmap.shape[1] and 0 <= y < heatmap.shape[0]:
            heatmap[y, x] += 1
    fig, ax = plt.subplots()
    ax.imshow(heatmap, cmap='hot')
    ax.set_title("Keypoint Density Heatmap (Image 1)")
    ax.axis('off')
    st.pyplot(fig)


st.title("🔍 SIFT-Based Image Comparison")

img1_file = st.file_uploader("Upload Image 1", type=["jpg", "png", "jpeg"])
img2_file = st.file_uploader("Upload Image 2", type=["jpg", "png", "jpeg"])

if img1_file and img2_file:
    image1 = load_image(img1_file)
    image2 = load_image(img2_file)
    title = st.text_input("Enter a title for the comparison:", "Image Match")
    
    if st.button("Analyze"):
        analyze_images(image1, image2, title)
