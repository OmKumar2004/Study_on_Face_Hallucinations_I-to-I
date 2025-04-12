import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import umap
import dlib
from PIL import Image
from io import BytesIO
from imutils import face_utils
import os

# Allow user to upload the .dat file
uploaded_predictor = st.file_uploader("Upload the shape_predictor_68_face_landmarks.dat file", type=["dat"])

# If the user uploads the .dat file
if uploaded_predictor is not None:
    # Save the uploaded file to disk
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    with open(predictor_path, "wb") as f:
        f.write(uploaded_predictor.getbuffer())

    # Load the Dlib predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

else:
    st.error("Please upload the 'shape_predictor_68_face_landmarks.dat' file.")
    st.stop()

# Rest of the code...
def load_image(uploaded_file, resize_dim=(300, 300)):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(resize_dim)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def plot_histograms(img, title_prefix):
    # RGB Histogram
    fig, ax = plt.subplots()
    colors = ('b', 'g', 'r')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax.plot(hist, color=col)
    ax.set_title(f'{title_prefix} - RGB Histogram')
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Grayscale Histogram
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fig, ax = plt.subplots()
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    ax.plot(hist, color='gray')
    ax.set_title(f'{title_prefix} - Grayscale Histogram')
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

def extract_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        return None, None
    shape = predictor(gray, rects[0])
    shape_np = face_utils.shape_to_np(shape)
    return rects[0], shape_np

def keypoints_per_feature(keypoints, landmarks):
    FACIAL_LANDMARKS_IDXS = {
        "mouth": (48, 68),
        "right_eyebrow": (17, 22),
        "left_eyebrow": (22, 27),
        "right_eye": (36, 42),
        "left_eye": (42, 48),
        "nose": (27, 36),
        "jaw": (0, 17)
    }
    feature_counts = {region: 0 for region in FACIAL_LANDMARKS_IDXS}
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        for region, (start, end) in FACIAL_LANDMARKS_IDXS.items():
            points = landmarks[start:end]
            if cv2.pointPolygonTest(np.array(points), (x, y), False) >= 0:
                feature_counts[region] += 1
                break
    return feature_counts

def plot_feature_counts(counts1, counts2):
    labels = list(counts1.keys())
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, [counts1[l] for l in labels], width, label='Image 1')
    ax.bar(x + width/2, [counts2[l] for l in labels], width, label='Image 2')
    ax.set_ylabel('Keypoints')
    ax.set_xlabel('Facial Feature Region')
    ax.set_title('Keypoints per Facial Feature')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    st.pyplot(fig)

def analyze_images(img1, img2, title):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    st.image(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB),
             caption=f"SIFT Matches - {title} | Good Matches: {len(good_matches)}",
             use_container_width=True)

    for label, kps in zip(['Image 1', 'Image 2'], [kp1, kp2]):
        sizes = [kp.size for kp in kps]
        angles = [kp.angle for kp in kps]

        fig, ax = plt.subplots()
        ax.hist(sizes, bins=30, alpha=0.7)
        ax.set_title(f'{label} - Keypoint Size Distribution')
        ax.set_xlabel("Keypoint Size")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.hist(angles, bins=36, alpha=0.7)
        ax.set_title(f'{label} - Keypoint Orientation Distribution')
        ax.set_xlabel("Angle (Degrees)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    if des1 is not None:
        sample_des = des1 if des1.shape[0] < 100 else des1[np.random.choice(des1.shape[0], 100, replace=False)]
        reducer = umap.UMAP(random_state=42)
        des_umap = reducer.fit_transform(sample_des)
        fig, ax = plt.subplots()
        ax.scatter(des_umap[:, 0], des_umap[:, 1], s=10, c='purple')
        ax.set_title("UMAP of SIFT Descriptors (Image 1)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        st.pyplot(fig)

    heatmap = np.zeros(img1_gray.shape)
    for kp in kp1:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= x < heatmap.shape[1] and 0 <= y < heatmap.shape[0]:
            heatmap[y, x] += 1
    fig, ax = plt.subplots()
    ax.imshow(heatmap, cmap='hot')
    ax.set_title("Keypoint Density Heatmap (Image 1)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('on')
    st.pyplot(fig)

    plot_histograms(img1, "Image 1")
    plot_histograms(img2, "Image 2")

    _, landmarks1 = extract_landmarks(img1)
    _, landmarks2 = extract_landmarks(img2)

    if landmarks1 is not None and landmarks2 is not None:
        feat1 = keypoints_per_feature(kp1, landmarks1)
        feat2 = keypoints_per_feature(kp2, landmarks2)
        plot_feature_counts(feat1, feat2)
    else:
        st.warning("Could not detect face/landmarks in one or both images for facial keypoint feature analysis.")

# UI
st.title("🔍 SIFT + Facial Feature-Based Image Comparison")

img1_file = st.file_uploader("Upload Image 1", type=["jpg", "png", "jpeg"])
img2_file = st.file_uploader("Upload Image 2", type=["jpg", "png", "jpeg"])

if img1_file and img2_file:
    image1 = load_image(img1_file)
    image2 = load_image(img2_file)
    title = st.text_input("Enter a title for the comparison:", "Image Match")

    if st.button("Analyze"):
        analyze_images(image1, image2, title)
