import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def save_debug_image(image, step_name, debug_dir="debug_outputs"):
    os.makedirs(debug_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(debug_dir, f"{timestamp}_{step_name}.png")
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return path

def detect_fingertips(segmented_hand_image, mask, debug=False):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    
    contour = max(contours, key=cv2.contourArea)

    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)

    fingertip_candidates = []
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, _, d = defects[i, 0]
            if d > 1000:
                fingertip_candidates.extend([tuple(approx[s][0]), tuple(approx[e][0])])

# ➕ Now correctly detected from the clean mask-based contour
    extreme_top = tuple(contour[contour[:, :, 1].argmin()][0])
    extreme_bottom = tuple(contour[contour[:, :, 1].argmax()][0])
    extreme_left = tuple(contour[contour[:, :, 0].argmin()][0])
    extreme_right = tuple(contour[contour[:, :, 0].argmax()][0])
    fingertip_candidates.extend([extreme_top, extreme_left, extreme_right])

    def euclidean_dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    clustered = []
    for pt in fingertip_candidates:
        if all(euclidean_dist(pt, c) > 20 for c in clustered):
            clustered.append(pt)

    if debug:
        temp_image = segmented_hand_image.copy()
        for pt in clustered:
            cv2.circle(temp_image, pt, 10, (255, 0, 255), -1)
        save_debug_image(temp_image, "fingertips_detected")

    return clustered

def filter_fingertips_by_position(fingertips, image_shape, vertical_percentile=0.5):
    """
    Remove fingertip points that are too low on the image (e.g., palm or wrist points).
    """
    height = image_shape[0]
    max_y = np.percentile([pt[1] for pt in fingertips], vertical_percentile * 100)
    filtered = [pt for pt in fingertips if pt[1] <= max_y]
    return filtered

def generate_nail_bounding_boxes(fingertips, contour, image, debug=False):
    if contour is None:
        return []

    M = cv2.moments(contour)
    cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)

    img_h, img_w = image.shape[:2]
    box_width = int(img_w * 0.1)
    box_height = int(img_h * 0.09)
    shift_x_distance = int(img_w * 0.08)
    shift_y_distance = int(img_h * 0.020)

    # Create boxes
    raw_boxes = []
    for pt in fingertips:
        direction = np.array([cx - pt[0], cy - pt[1]])
        norm = np.linalg.norm(direction)
        if norm != 0:
            direction = direction / norm
        center = (pt[0] + int(direction[0] * shift_x_distance), pt[1] + int(direction[1] * shift_y_distance))
        top_left = (center[0] - box_width // 2, center[1] - box_height // 2)
        bottom_right = (center[0] + box_width // 2, center[1] + box_height // 2)
        raw_boxes.append((top_left, bottom_right))

    # New merging logic
    def merge_close_boxes(bboxes, threshold=int(min(img_w, img_h) * 0.10)):
        def center(box): return ((box[0][0] + box[1][0]) // 2, (box[0][1] + box[1][1]) // 2)
        def dist(p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))

        centers = [center(b) for b in bboxes]
        used = [False] * len(bboxes)
        merged = []

        for i in range(len(bboxes)):
            if used[i]: continue
            group = [bboxes[i]]
            used[i] = True
            for j in range(i + 1, len(bboxes)):
                if not used[j] and dist(centers[i], centers[j]) < threshold:
                    group.append(bboxes[j])
                    used[j] = True
            # Average the box coordinates
            xs1, ys1 = zip(*[b[0] for b in group])
            xs2, ys2 = zip(*[b[1] for b in group])
            merged.append(((int(np.mean(xs1)), int(np.mean(ys1))), (int(np.mean(xs2)), int(np.mean(ys2)))))

        return merged

    final_boxes = merge_close_boxes(raw_boxes)

    # Optional debug output
    if debug:
        debug_canvas = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
        original_overlay = image.copy()

        for top_left, bottom_right in final_boxes:
            cv2.rectangle(debug_canvas, top_left, bottom_right, (255, 0, 0), 2)
            cv2.rectangle(original_overlay, top_left, bottom_right, (255, 0, 0), 2)

        save_debug_image(debug_canvas, "bounding_boxes_white_bg")
        save_debug_image(original_overlay, "bounding_boxes_on_original")

    return final_boxes

def crop_bounding_boxes(image, bounding_boxes, debug=False):
    cropped_images = []
    for (top_left, bottom_right) in bounding_boxes:
        x1, y1 = top_left
        x2, y2 = bottom_right
        cropped = image[y1:y2, x1:x2]
        cropped_images.append(cropped)

    if debug and cropped_images:
        cols = 4
        rows = (len(cropped_images) + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axs = axs.flatten()
        for idx in range(rows * cols):
            axs[idx].axis('off')
            if idx < len(cropped_images):
                axs[idx].imshow(cropped_images[idx])
                axs[idx].set_title(f"Crop {idx+1}")
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("debug_outputs", exist_ok=True)
        plt.savefig(f"debug_outputs/{timestamp}_nail_crops_grid.png")
        plt.close()

    return cropped_images

def filter_crops_by_brightness(cropped_images, brightness_threshold=240, max_white_ratio=0.7):
    """
    Remove crops that are overly bright (non-nail, background, or palm).
    """
    selected = []
    for img in cropped_images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        white_ratio = np.sum(gray > brightness_threshold) / gray.size
        if white_ratio < max_white_ratio:
            selected.append(img)
    return selected

def select_three_nails_with_least_background(cropped_images, background_threshold=240, background_ratio_cutoff=0.7, debug=False):
    background_ratios = []

    for img in cropped_images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        white_pixels = np.sum(gray > background_threshold)
        total_pixels = gray.size
        background_ratio = white_pixels / total_pixels
        background_ratios.append(background_ratio)

    sorted_indices = sorted(range(len(cropped_images)), key=lambda i: background_ratios[i])
    filtered_indices = [i for i in sorted_indices if background_ratios[i] < background_ratio_cutoff]
    selected_images = [cropped_images[i] for i in filtered_indices[:3]]

    if debug:
        cols = 3
        fig, axs = plt.subplots(1, cols, figsize=(15, 5))
        for i in range(cols):
            axs[i].axis('off')
            if i < len(selected_images):
                axs[i].imshow(selected_images[i])
                axs[i].set_title(f"Selected {i+1}")
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("debug_outputs", exist_ok=True)
        plt.savefig(f"debug_outputs/{timestamp}_selected_nails_grid.png")
        plt.close()

    return selected_images

def find_brightest_patch(image, patch_size=(50, 50), stride=10):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    ph, pw = patch_size
    max_mean = -1
    best_patch = (0, 0)

    for y in range(0, h - ph + 1, stride):
        for x in range(0, w - pw + 1, stride):
            patch = gray[y:y + ph, x:x + pw]
            patch_mean = np.mean(patch)
            if patch_mean > max_mean:
                max_mean = patch_mean
                best_patch = (y, x)

    y, x = best_patch
    return image[y:y + ph, x:x + pw]  # Return the RGB patch

def compute_normalized_rgb_from_reference_region_fixed(nail_images, original_image, ref_box_size=(50, 50), debug=False):
    height, width, _ = original_image.shape
    # box_h, box_w = ref_box_size
    # ref_region = original_image[height - box_h:height, 0:box_w]
    ref_region = find_brightest_patch(original_image, patch_size=(int(height*.015), int(width*.015)))

    white_ref_median = {
        color: np.median(ref_region[:, :, chan].ravel())
        for chan, color in enumerate("RGB")
    }

    feature_dict = {}
    for i, img in enumerate(nail_images[:3]):
        for chan, color in enumerate("RGB"):
            mean_val = np.median(img[:, :, chan])
            norm_val = mean_val / white_ref_median[color]
            feature_dict[f'NAIL_{i+1}_{color}_mean'] = norm_val

    if debug:
        save_debug_image(ref_region, "white_reference_region")

    return pd.DataFrame([feature_dict])     

def extract_features_from_image(image_bytes, model, debug=False):

    def save_debug_image(image, step_name, debug_dir="debug_outputs"):
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(debug_dir, f"{timestamp}_{step_name}.png")
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return path

    # Decode and convert to RGB
    image_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if debug:
        save_debug_image(image, "input_image")

    # 🔍 Step 2: YOLOv8 Nail Detection
    results = model.predict(source=image, conf=0.3, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)

    if len(boxes) == 0:
        print("⚠️ No nails detected.")
        return pd.DataFrame()

    elif len(boxes) < 3:
        print(f"⚠️ Only {len(boxes)} nails detected, duplicating to make 3.")
        while len(boxes) < 3:
            boxes = np.vstack([boxes, boxes[-1]])

    # Sort and take top 3 boxes (you can sort by confidence or position)
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))[:3]  # Top 3 by vertical position (y1)

    # Convert to (top_left, bottom_right)
    bounding_boxes = [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in boxes]

    if debug:
        image_with_boxes = image.copy()
        for (x1, y1), (x2, y2) in bounding_boxes:
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        save_debug_image(image_with_boxes, "yolo_detections")

    # 🔍 Step 3: Crop and filter
    cropped_nails = crop_bounding_boxes(image, bounding_boxes, debug=debug)
    cropped_nails = filter_crops_by_brightness(cropped_nails)

    # 🔍 Step 4: Pick best nails
    best_nails = select_three_nails_with_least_background(cropped_nails, debug=debug)

    # 🔍 Step 5: Normalize RGB
    features = compute_normalized_rgb_from_reference_region_fixed(best_nails, image, debug=debug)

    return features