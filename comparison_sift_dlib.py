import cv2
import dlib
import numpy as np
import glob
import os

# Directories for the original, generated images and the output folder.
orig_dir = 'original_images'
gen_dir = os.path.join('results', 'codeformer_32')
output_dir = os.path.join('comparison_results', 'codeformer_32_results', 'dlib')
os.makedirs(output_dir, exist_ok=True)

# Gather image paths from original_images folder (both .jpg and .png)
orig_files = sorted(glob.glob(os.path.join(orig_dir, "*.jpg")) + glob.glob(os.path.join(orig_dir, "*.png")))
num_images = min(5, len(orig_files))

# Initialize SIFT detector.
sift = cv2.SIFT_create()

# Initialize dlib's face detector and landmark predictor.
face_detector = dlib.get_frontal_face_detector()
landmark_model_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(landmark_model_path)

def get_landmarks(image):
    """Detect facial landmarks using dlib's detector."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    if len(faces) == 0:
        return None
    # Assume the first detected face is the subject.
    shape = predictor(gray, faces[0])
    # Convert the dlib shape into a NumPy array (68,2)
    coords = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.int32)
    return coords

def annotate_differences(orig_img, gen_img, kp_orig, kp_gen, good_matches, landmarks_orig, landmarks_gen):
    """
    Create a visualization image showing:
      - On the original image: SIFT good matches (green) and the facial landmarks labeled.
      - On the generated image: SIFT good matches (green), unmatched keypoints (red), and facial landmarks.
    """
    orig_vis = orig_img.copy()
    gen_vis = gen_img.copy()

    # Draw landmark points with their index (for human interpretation) on original image.
    if landmarks_orig is not None:
        for i, (x, y) in enumerate(landmarks_orig):
            cv2.circle(orig_vis, (x, y), 2, (255, 0, 0), -1)
            cv2.putText(orig_vis, f"{i}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    if landmarks_gen is not None:
        for i, (x, y) in enumerate(landmarks_gen):
            cv2.circle(gen_vis, (x, y), 2, (255, 0, 0), -1)
            cv2.putText(gen_vis, f"{i}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Create a set of indices for good matches in the generated image.
    matched_gen_indices = {m.trainIdx for m in good_matches}
    
    # Draw good match keypoints (green) on original and generated images.
    for m in good_matches:
        center_orig = tuple(np.int32(kp_orig[m.queryIdx].pt))
        cv2.circle(orig_vis, center_orig, 3, (0, 255, 0), -1)
        center_gen = tuple(np.int32(kp_gen[m.trainIdx].pt))
        cv2.circle(gen_vis, center_gen, 3, (0, 255, 0), -1)
    
    # Mark unmatched keypoints in the generated image (red).
    for idx, kp in enumerate(kp_gen):
        if idx not in matched_gen_indices:
            center = tuple(np.int32(kp.pt))
            cv2.circle(gen_vis, center, 3, (0, 0, 255), -1)
    
    # Combine images side-by-side.
    combined = np.hstack((orig_vis, gen_vis))
    return combined

def find_nearest_kp_for_landmark(landmark, keypoints, max_distance=30):
    """
    Find the SIFT keypoint in keypoints that is nearest to the given landmark.
    Returns index of the keypoint and the distance, or (None, None) if none found within max_distance.
    """
    landmark = np.array(landmark)
    min_dist = float("inf")
    best_idx = None
    for i, kp in enumerate(keypoints):
        pt = np.array(kp.pt)
        dist = np.linalg.norm(landmark - pt)
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    if min_dist <= max_distance:
        return best_idx, min_dist
    else:
        return None, None

for idx, orig_path in enumerate(orig_files[:num_images]):
    base_name_no_ext = os.path.splitext(os.path.basename(orig_path))[0]
    gen_file_name = base_name_no_ext + "_restored.png"
    gen_path = os.path.join(gen_dir, gen_file_name)
    print(f"Processing pair: Original: {orig_path} | Generated: {gen_path}")
    
    if not os.path.exists(gen_path):
        print(f"File not found: {gen_path}. Skipping {base_name_no_ext}...")
        continue

    img_original = cv2.imread(orig_path)
    img_generated = cv2.imread(gen_path)
    
    if img_original is None or img_generated is None:
        print(f"Error loading one of the images for {base_name_no_ext}. Skipping...")
        continue
    
    gray_orig = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    gray_gen = cv2.cvtColor(img_generated, cv2.COLOR_BGR2GRAY)

    kp_orig, des_orig = sift.detectAndCompute(gray_orig, None)
    kp_gen, des_gen = sift.detectAndCompute(gray_gen, None)
    
    if des_orig is None or des_gen is None:
        print(f"No descriptors found for {base_name_no_ext}. Skipping...")
        continue

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_orig, des_gen)
    matches = sorted(matches, key=lambda m: m.distance)
    distance_threshold = 150
    good_matches = [m for m in matches if m.distance < distance_threshold]
    
    # Get facial landmarks using dlib.
    landmarks_orig = get_landmarks(img_original)
    landmarks_gen = get_landmarks(img_generated)

    # Prepare a machine-level description by associating landmarks with nearest SIFT keypoints.
    landmark_report_lines = []
    landmark_names = {
        0: "Jaw start", 1: "Jaw", 2: "Jaw", 3: "Jaw", 4: "Jaw", 5: "Jaw", 6: "Jaw", 7: "Jaw",
        8: "Chin", 9: "Jaw", 10: "Jaw", 11: "Jaw", 12: "Jaw", 13: "Jaw", 14: "Jaw", 15: "Jaw", 16: "Jaw end",
        17: "Left eyebrow", 18: "Left eyebrow", 19: "Left eyebrow", 20: "Left eyebrow", 21: "Left eyebrow",
        22: "Right eyebrow", 23: "Right eyebrow", 24: "Right eyebrow", 25: "Right eyebrow", 26: "Right eyebrow",
        27: "Nose bridge", 28: "Nose bridge", 29: "Nose bridge", 30: "Nose tip", 31: "Nose", 32: "Nose", 33: "Nose", 34: "Nose", 35: "Nose",
        36: "Left eye corner", 37: "Left eye", 38: "Left eye", 39: "Left eye", 40: "Left eye", 41: "Left eye",
        42: "Right eye corner", 43: "Right eye", 44: "Right eye", 45: "Right eye corner", 46: "Right eye", 47: "Right eye",
        48: "Left mouth corner", 49: "Outer lip", 50: "Outer lip", 51: "Outer lip", 52: "Outer lip", 53: "Outer lip", 54: "Right mouth corner", 55: "Outer lip", 56: "Outer lip", 57: "Outer lip", 58: "Outer lip", 59: "Outer lip",
        60: "Inner lip", 61: "Inner lip", 62: "Inner lip", 63: "Inner lip", 64: "Inner lip", 65: "Inner lip", 66: "Inner lip", 67: "Inner lip"
    }
    
    landmark_report_lines.append(f"Semantic Landmark Comparison for {base_name_no_ext}:")
    if landmarks_orig is None or landmarks_gen is None:
        landmark_report_lines.append("  Facial landmarks not detected in one of the images.")
    else:
        # For each landmark in the original image that is of interest,
        # find its nearest SIFT keypoint in both images.
        for idx_l, landmark in enumerate(landmarks_orig):
            if idx_l not in landmark_names:
                continue  # Process only the landmarks we have names for.
            name = landmark_names[idx_l]
            kp_idx_orig, dist_orig = find_nearest_kp_for_landmark(landmark, kp_orig)
            kp_idx_gen, dist_gen = find_nearest_kp_for_landmark(landmark, kp_gen)
            if kp_idx_orig is None:
                landmark_report_lines.append(f"  {name}: Not detected in original image (landmark might be off).")
            elif kp_idx_gen is None:
                landmark_report_lines.append(f"  {name}: Missing in generated image (potential hallucination).")
            else:
                # Check if the keypoints correspond via SIFT matching.
                # We check if there's a good match linking these nearest keypoints.
                corresponding_matches = [m for m in good_matches if m.queryIdx == kp_idx_orig and m.trainIdx == kp_idx_gen]
                if corresponding_matches:
                    landmark_report_lines.append(f"  {name}: Consistent (match distance: {corresponding_matches[0].distance:.2f}).")
                else:
                    landmark_report_lines.append(f"  {name}: Inconsistent between images (distances: orig {dist_orig:.2f}, gen {dist_gen:.2f}).")
    
    report_text = "\n".join(landmark_report_lines)
    print(report_text + "\n")
    report_file = os.path.join(output_dir, f"semantic_report_{base_name_no_ext}.txt")
    with open(report_file, "w") as f:
        f.write(report_text)
    
    # Generate annotated visualization combining SIFT match data and landmark displays.
    diff_vis = annotate_differences(img_original, img_generated, kp_orig, kp_gen, good_matches,
                                    landmarks_orig, landmarks_gen)
    output_image_path = os.path.join(output_dir, f"diff_{base_name_no_ext}.png")
    cv2.imwrite(output_image_path, diff_vis)

print("Processing complete. Check the output folder for images and semantic reports.")
