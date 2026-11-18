"""
Machine Vision (UFMFRR-15-M) - Group Portfolio Project
Unified End-to-End Object Counting Pipeline: Stages 1 to 5

Primary Contributor: Jakub Wrzyszcz
"""

import os
import glob
import time
import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    mean_absolute_error, mean_squared_error
)


# ==============================================================================
# DATASET UTILITIES & GROUND TRUTH
# ==============================================================================

def resolve_dataset_files(target="apple"):
    """Finds image file paths for either MinneApple or Resistor Dataset."""
    if target == "apple":
        patterns = [
            "minneapple-1-1/train/*.jpg",
            "/content/minneapple-1-1/train/*.jpg",
            "**/minneapple-1-1/train/*.jpg",
            "**/train/*.jpg",
        ]
    else:
        patterns = [
            "Resistor Dataset/*.jpg",
            "Resistor Dataset/*.png",
            "Resistor Dataset/*.jpeg",
            "/content/Resistor Dataset/*.jpg",
            "**/Resistor Dataset/*.jpg",
        ]

    for p in patterns:
        files = glob.glob(p, recursive=True)
        if files:
            return sorted(files)
    return []


def load_normalized_image(img_path, max_dim=1280):
    """Loads an image, standardizes max dimension, and converts BGR to RGB."""
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image at path: {img_path}")
    h, w = bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = float(max_dim) / float(max(h, w))
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_coco_annotations(json_path="minneapple-1-1/train/_annotations.coco.json"):
    """Loads COCO ground-truth [x, y, w, h] bounding boxes mapped by filename."""
    if not os.path.exists(json_path):
        alt = glob.glob("**/_annotations.coco.json", recursive=True)
        if alt:
            json_path = alt[0]
        else:
            return {}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
        img_id_to_file = {img["id"]: img["file_name"] for img in coco.get("images", [])}
        file_to_boxes = {}
        for ann in coco.get("annotations", []):
            img_id = ann.get("image_id")
            fname = img_id_to_file.get(img_id)
            if fname:
                file_to_boxes.setdefault(fname, []).append(ann.get("bbox", []))
        return file_to_boxes
    except Exception:
        return {}


# ==============================================================================
# STAGE 1: UNIVERSAL RADIOMETRIC & SPATIAL PREPROCESSING (UNITS 2, 4, 5)
# ==============================================================================

def compute_intensity(rgb_img):
    """Computes Unit 2 Intensity: I = (R + G + B) / 3."""
    return np.mean(rgb_img.astype(np.float32), axis=2).astype(np.uint8)


def preprocess_image(rgb_img, clahe_clip=2.0, grid_size=(8, 8), median_k=3, gauss_sigma=0.8):
    """Applies LAB L-channel CLAHE, RGB median/Gaussian denoising, and channel extraction."""
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l_raw, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=grid_size)
    l_eq = clahe.apply(l_raw)
    eq_rgb = cv2.cvtColor(cv2.merge([l_eq, a_chan, b_chan]), cv2.COLOR_LAB2RGB)

    denoised_rgb = cv2.medianBlur(eq_rgb, median_k)
    clean_rgb = cv2.GaussianBlur(denoised_rgb, (3, 3), gauss_sigma)

    hsv = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    i_chan = compute_intensity(clean_rgb)

    return clean_rgb, h, s, v, i_chan, l_raw, l_eq


# ==============================================================================
# STAGE 2: UNIVERSAL FOREGROUND SEGMENTATION & BENCHMARKING (UNITS 3, 4)
# ==============================================================================

def segment_foreground(rgb_img, method="auto", min_area=80):
    """Scene-adaptive foreground segmentation followed by 5x5 closing and hole filling."""
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0]

    border = np.concatenate([l_chan[0, :], l_chan[-1, :], l_chan[:, 0], l_chan[:, -1]])
    bg_L = float(np.median(border))

    use_adaptive = (method == "adaptive") or (method == "auto" and bg_L > 150.0)

    if use_adaptive:
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        raw_binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 15
        )
        debug_info = {"method": "adaptive_gaussian", "bg_L": bg_L}
    else:
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        a_chan = lab[:, :, 1].astype(np.float32)
        b_chan = lab[:, :, 2].astype(np.float32)
        s_chan = hsv[:, :, 1].astype(np.float32)

        fruit_chroma = np.maximum(a_chan - 124.0, 0.0) + np.maximum(b_chan - 138.0, 0.0)
        saliency = fruit_chroma * (s_chan / 255.0)
        saliency_norm = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        thresh_val, raw_binary = cv2.threshold(saliency_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        debug_info = {"method": "otsu_chroma_fusion", "thresh_val": thresh_val, "bg_L": bg_L}

    kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(raw_binary, cv2.MORPH_CLOSE, kernel_5x5, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    clean_mask = np.zeros_like(filled)
    valid_contours = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(clean_mask, [c], -1, 255, thickness=cv2.FILLED)
            valid_contours.append(c)

    return clean_mask, debug_info, valid_contours


def benchmark_clustering(rgb_img, k=3):
    """Clusters pixels in 3D (S, a*, b*) space using cv2.kmeans for benchmarking."""
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    s_chan = hsv[:, :, 1]
    a_chan = lab[:, :, 1]
    b_chan = lab[:, :, 2]

    h, w = rgb_img.shape[:2]
    features = np.column_stack([s_chan.ravel(), a_chan.ravel(), b_chan.ravel()]).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)

    _, labels, centers = cv2.kmeans(features, k, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS)
    fg_cluster_idx = int(np.argmax(centers[:, 0] + centers[:, 1]))
    cluster_mask = (labels.reshape((h, w)) == fg_cluster_idx).astype(np.uint8) * 255

    return cluster_mask, centers


# ==============================================================================
# STAGE 3: SHAPE-ADAPTIVE MORPHOLOGICAL PROPOSAL GENERATION (UNIT 4)
# ==============================================================================

def extract_proposals(rgb_img, binary_mask, target_size=(64, 64), min_area=50):
    """
    Universal Stage 3 Proposal Generator:
      - Aspect Ratio < 3.0 (Radial): Distance Transform + Watershed cluster splitting.
      - Aspect Ratio >= 3.0 (Axial): Affine pose alignment, lead stripping, and 1D sliding window.
    """
    h_img, w_img = rgb_img.shape[:2]
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    s_chan = hsv[:, :, 1]

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops, masks, metadata, conf_scores = [], [], [], []

    for idx, c in enumerate(contours):
        if cv2.contourArea(c) < min_area:
            continue

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        e1 = box[1] - box[0]
        e2 = box[2] - box[1]
        l1 = float(np.linalg.norm(e1))
        l2 = float(np.linalg.norm(e2))

        major_len = max(l1, l2)
        minor_len = min(l1, l2)
        aspect_ratio = major_len / (minor_len + 1e-6)

        # Case A: Radial / Compact Instances (e.g., orchard apples)
        if aspect_ratio < 3.0:
            comp_mask = np.zeros_like(binary_mask)
            cv2.drawContours(comp_mask, [c], -1, 255, thickness=cv2.FILLED)

            dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            max_d = dist.max()
            if max_d < 3.0:
                continue

            _, sure_fg = cv2.threshold(dist, 0.35 * max_d, 255, cv2.THRESH_BINARY)
            sure_fg = sure_fg.astype(np.uint8)
            sure_bg = cv2.dilate(comp_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
            unknown = cv2.subtract(sure_bg, sure_fg)

            num_markers, markers = cv2.connectedComponents(sure_fg)
            if num_markers <= 1:
                M = cv2.moments(comp_mask)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    tx, ty, tw, th = cv2.boundingRect(c)
                    side = int(np.clip(max(major_len * 1.1, 48), 48, 120))
                    hs = side // 2
                    x1, x2 = max(0, cx - hs), min(w_img, cx + hs)
                    y1, y2 = max(0, cy - hs), min(h_img, cy + hs)
                    p_rgb = rgb_img[y1:y2, x1:x2]
                    p_mask = comp_mask[y1:y2, x1:x2]
                    if p_rgb.shape[0] >= 4 and p_rgb.shape[1] >= 4:
                        perim = cv2.arcLength(c, True)
                        circ = (4.0 * np.pi * float(cv2.contourArea(c))) / ((perim ** 2) + 1e-6)
                        score = circ * float(np.mean(s_chan[y1:y2, x1:x2])) * float(max_d)

                        crops.append(cv2.resize(p_rgb, target_size, interpolation=cv2.INTER_AREA))
                        masks.append(cv2.resize(p_mask, target_size, interpolation=cv2.INTER_NEAREST))
                        conf_scores.append(score)
                        metadata.append({
                            "type": "radial_single",
                            "centroid": (cx, cy),
                            "bbox": [tx, ty, tw, th],  # Tight contour bbox for accurate COCO IoU matching
                            "aspect_ratio": aspect_ratio,
                            "area": int(M["m00"]),
                        })
            else:
                markers = markers + 1
                markers[unknown == 255] = 0
                markers = cv2.watershed(rgb_img.copy(), markers)
                for label in np.unique(markers):
                    if label < 2:
                        continue
                    inst_m = (markers == label).astype(np.uint8) * 255
                    inst_area = np.sum(inst_m > 0)
                    if inst_area < min_area:
                        continue
                    M = cv2.moments(inst_m)
                    if M["m00"] == 0:
                        continue
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    tx, ty, tw, th = cv2.boundingRect(inst_m)
                    side = int(np.clip(max(tw, th), 48, 120))
                    hs = side // 2
                    x1, x2 = max(0, cx - hs), min(w_img, cx + hs)
                    y1, y2 = max(0, cy - hs), min(h_img, cy + hs)
                    p_rgb = rgb_img[y1:y2, x1:x2]
                    p_mask = inst_m[y1:y2, x1:x2]
                    if p_rgb.shape[0] >= 4 and p_rgb.shape[1] >= 4:
                        inst_cnts, _ = cv2.findContours(inst_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if inst_cnts:
                            perim = cv2.arcLength(inst_cnts[0], True)
                            circ = (4.0 * np.pi * float(inst_area)) / ((perim ** 2) + 1e-6)
                        else:
                            circ = 0.5
                        score = circ * float(np.mean(s_chan[y1:y2, x1:x2])) * float(np.max(dist[y1:y2, x1:x2]))

                        crops.append(cv2.resize(p_rgb, target_size, interpolation=cv2.INTER_AREA))
                        masks.append(cv2.resize(p_mask, target_size, interpolation=cv2.INTER_NEAREST))
                        conf_scores.append(score)
                        metadata.append({
                            "type": "radial_watershed",
                            "centroid": (cx, cy),
                            "bbox": [tx, ty, tw, th],  # Tight Watershed instance bbox for COCO IoU matching
                            "aspect_ratio": aspect_ratio,
                            "area": int(inst_area),
                        })

        # Case B: Axial / Elongated Instances (e.g., cylindrical resistors)
        else:
            long_vec = e1 if l1 >= l2 else e2
            if major_len < 80 or minor_len < 8:
                continue

            angle = float(np.degrees(np.arctan2(long_vec[1], long_vec[0])))
            cx, cy = rect[0]
            orig_x, orig_y, orig_w, orig_h = cv2.boundingRect(c)

            M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            rot_img = cv2.warpAffine(
                rgb_img, M_rot, (w_img, h_img), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            comp_mask = np.zeros_like(binary_mask)
            cv2.drawContours(comp_mask, [c], -1, 255, thickness=cv2.FILLED)
            rot_mask = cv2.warpAffine(comp_mask, M_rot, (w_img, h_img), flags=cv2.INTER_NEAREST)

            x1 = int(max(0, cx - major_len / 2))
            x2 = int(min(w_img, cx + major_len / 2))
            y1 = int(max(0, cy - minor_len * 0.75))
            y2 = int(min(h_img, cy + minor_len * 0.75))

            c_crop = rot_img[y1:y2, x1:x2]
            c_mask = rot_mask[y1:y2, x1:x2]
            cw = c_crop.shape[1]
            if cw < 50:
                continue

            col_thick = np.sum(c_mask > 0, axis=0)
            smooth_thick = np.convolve(col_thick, np.ones(21) / 21.0, mode="same")
            c_start, c_end = int(cw * 0.25), int(cw * 0.75)
            if c_end <= c_start:
                continue

            center_pk = c_start + int(np.argmax(smooth_thick[c_start:c_end]))
            max_th = smooth_thick[center_pk]
            if max_th < 15:
                continue

            bx1 = center_pk
            while bx1 > max(0, c_start - 30) and smooth_thick[bx1] > 0.55 * max_th:
                bx1 -= 1
            bx2 = center_pk
            while bx2 < min(cw - 1, c_end + 30) and smooth_thick[bx2] > 0.55 * max_th:
                bx2 += 1

            body_crop = c_crop[:, bx1:bx2]
            body_m = c_mask[:, bx1:bx2]

            row_thick = np.sum(body_m > 0, axis=1)
            if len(row_thick) > 0 and row_thick.max() > 0:
                act_rows = np.where(row_thick > 0.4 * row_thick.max())[0]
                if len(act_rows) > 10:
                    body_crop = body_crop[act_rows[0]:act_rows[-1], :]
                    body_m = body_m[act_rows[0]:act_rows[-1], :]

            bh, bw = body_crop.shape[:2]
            if bw < 25 or bh < 12:
                continue

            c_strip = body_crop[int(bh * 0.2):int(bh * 0.8), :]
            gray_strip = cv2.cvtColor(c_strip, cv2.COLOR_RGB2GRAY)
            sobel_x = cv2.Sobel(gray_strip, cv2.CV_64F, 1, 0, ksize=3)
            sobel_energy = np.mean(np.abs(sobel_x), axis=0)

            win_w = max(12, int(bw * 0.12))
            step = max(6, int(win_w * 0.5))

            # Estimate baseline ceramic body LAB color from the end caps
            edge_strip = np.concatenate([body_crop[:, :max(4, win_w // 2)], body_crop[:, -max(4, win_w // 2):]], axis=1)
            base_lab = np.median(cv2.cvtColor(edge_strip, cv2.COLOR_RGB2LAB).reshape(-1, 3), axis=0)

            for wx in range(0, bw - win_w, step):
                band_p = body_crop[:, wx:wx + win_w]
                band_m = body_m[:, wx:wx + win_w]
                g_score = float(np.mean(sobel_energy[wx:wx + win_w])) if len(sobel_energy) > wx else 1.0

                crops.append(cv2.resize(band_p, target_size, interpolation=cv2.INTER_AREA))
                masks.append(cv2.resize(band_m, target_size, interpolation=cv2.INTER_NEAREST))
                conf_scores.append(float(bw * bh) + g_score)
                metadata.append({
                    "type": "axial_band",
                    "resistor_id": idx,
                    "window_x": wx,
                    "rel_pos": float(wx / max(1, bw)),
                    "sobel_score": g_score,
                    "aspect_ratio": aspect_ratio,
                    "body_shape": (bh, bw),
                    "base_lab": base_lab,
                    "bbox": [orig_x, orig_y, orig_w, orig_h],
                })

    return crops, masks, metadata


# ==============================================================================
# STAGE 4: UNIFIED 95D FEATURE EXTRACTION & PCA (UNITS 2, 4, 6, 7)
# ==============================================================================

def extract_95d_features(crop_rgb, crop_mask=None):
    """Extracts 32 foreground-masked colour features, 59 nri_uniform LBP bins, and 4 shape metrics."""
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    h_ch, s_ch, _ = cv2.split(hsv)
    _, a_ch, b_ch = cv2.split(lab)
    i_ch = compute_intensity(crop_rgb)

    if crop_mask is not None and np.any(crop_mask > 0):
        fg_mask = (crop_mask > 0)
    else:
        fg_mask = np.ones(h_ch.shape, dtype=bool)

    channels = [h_ch, s_ch, a_ch, b_ch, i_ch]
    moments = []
    for ch in channels:
        vals = ch[fg_mask].astype(np.float32)
        if len(vals) < 4:
            vals = ch.ravel().astype(np.float32)
        moments.extend([
            float(np.mean(vals)),
            float(np.std(vals)),
            float(stats.skew(vals)) if len(vals) > 2 else 0.0,
            float(stats.kurtosis(vals)) if len(vals) > 3 else 0.0,
        ])

    h_vals = h_ch[fg_mask]
    s_vals = s_ch[fg_mask]
    a_vals = a_ch[fg_mask]
    if len(h_vals) == 0:
        h_vals, s_vals, a_vals = h_ch.ravel(), s_ch.ravel(), a_ch.ravel()

    h_hist, _ = np.histogram(h_vals, bins=4, range=(0, 180), density=True)
    s_hist, _ = np.histogram(s_vals, bins=4, range=(0, 256), density=True)
    a_hist, _ = np.histogram(a_vals, bins=4, range=(0, 256), density=True)
    colour_feats = np.concatenate([moments, h_hist, s_hist, a_hist]).astype(np.float32)
    colour_feats = np.nan_to_num(colour_feats, nan=0.0, posinf=0.0, neginf=0.0)

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    lbp_map = local_binary_pattern(gray, P=8, R=1, method="nri_uniform")
    texture_feats, _ = np.histogram(lbp_map.ravel(), bins=59, range=(0, 59), density=True)
    texture_feats = texture_feats.astype(np.float32)

    if crop_mask is None:
        _, crop_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        perimeter = float(cv2.arcLength(c, True))
        circularity = (4.0 * np.pi * area) / ((perimeter ** 2) + 1e-6)
        hull = cv2.convexHull(c)
        solidity = area / (float(cv2.contourArea(hull)) + 1e-6)
        _, _, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / float(h + 1e-6)
        extent = area / float(w * h + 1e-6)
    else:
        circularity, solidity, aspect_ratio, extent = 0.0, 0.0, 1.0, 0.0

    shape_feats = np.array([circularity, solidity, aspect_ratio, extent], dtype=np.float32)

    vector_95d = np.concatenate([colour_feats, texture_feats, shape_feats])
    assert len(vector_95d) == 95, f"Expected 95D vector, got {len(vector_95d)}D"
    return vector_95d


def fit_scaler_and_pca(X_train, X_val=None, n_components=0.95):
    """Fits StandardScaler and PCA strictly on the training split."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=n_components, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_scaled)

    X_val_pca = None
    if X_val is not None and len(X_val) > 0:
        X_val_pca = pca.transform(scaler.transform(X_val))

    return scaler, pca, X_train_pca, X_val_pca


# ==============================================================================
# STAGE 5: MACHINE LEARNING, NON-MAXIMUM SUPPRESSION & COUNTING EVALUATION
# ==============================================================================

def compute_box_iou(boxA, boxB):
    """Computes Intersection-over-Union (IoU) between two [x, y, w, h] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    return inter / float(areaA + areaB - inter + 1e-6)


def apply_2d_nms(boxes, scores, iou_thresh=0.35):
    """2D Non-Maximum Suppression to eliminate overlapping detections."""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def apply_1d_nms(slices_meta, predicted_labels, confidences, bare_class="Bare_Body"):
    """1D Non-Maximum Suppression grouping consecutive axial slices into discrete bands."""
    if not predicted_labels:
        return []

    sorted_indices = sorted(range(len(slices_meta)), key=lambda i: slices_meta[i].get("window_x", i))
    bands = []
    current_label = None
    best_conf = -1.0
    best_idx = None

    for idx in sorted_indices:
        lbl = predicted_labels[idx]
        conf = confidences[idx]
        if lbl == bare_class:
            if current_label is not None and current_label != bare_class:
                bands.append((current_label, best_conf, slices_meta[best_idx]))
                current_label = None
                best_conf = -1.0
                best_idx = None
        else:
            if lbl == current_label:
                if conf > best_conf:
                    best_conf = conf
                    best_idx = idx
            else:
                if current_label is not None and current_label != bare_class:
                    bands.append((current_label, best_conf, slices_meta[best_idx]))
                current_label = lbl
                best_conf = conf
                best_idx = idx

    if current_label is not None and current_label != bare_class:
        bands.append((current_label, best_conf, slices_meta[best_idx]))

    return bands


def count_resistor_bands(bands):
    """
    Counts the number of colour bands of each colour present on each resistor
    in accordance with the assessment specification.
    Returns:
        valid_bands: list of cleaned EIA colour strings
        counts_dict: dictionary of counts per colour and total bands
    """
    EIA_COLOURS = {
        "Black", "Brown", "Red", "Orange", "Yellow",
        "Green", "Blue", "Violet", "Gray", "White", "Gold", "Silver"
    }
    valid_bands = [str(b) for b in bands if str(b) in EIA_COLOURS]
    counts_dict = {c: valid_bands.count(c) for c in sorted(set(valid_bands))}
    counts_dict["Total Bands"] = len(valid_bands)
    return valid_bands, counts_dict


def classify_axial_slice(crop_rgb, base_lab=None):
    """Assigns reference EIA colour labels (12 colours + Bare_Body) to axial slices."""
    w_lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    w_hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    med_lab = np.median(w_lab.reshape(-1, 3), axis=0)
    med_hsv = np.median(w_hsv.reshape(-1, 3), axis=0)
    med_rgb = np.median(crop_rgb.reshape(-1, 3), axis=0)

    if base_lab is not None:
        delta_e = float(np.linalg.norm(med_lab - base_lab))
        if delta_e < 18.0:
            return "Bare_Body", 1.0 - (delta_e / 25.0)

    h, s, v = med_hsv
    L, a, b = med_lab
    I = float(np.mean(med_rgb))

    if b > 140 and a > 126 and s > 35 and 60 < I < 145:
        if 15 <= h <= 35:
            return "Gold", float(b / 255.0)
    if a > 136 and (h <= 10 or h >= 165) and s > 55:
        return "Red", float(a / 255.0)
    if 10 < h <= 25 and s > 75 and a > 132:
        return "Orange", float(s / 255.0)
    if 25 < h <= 40 and b > 142 and s > 65:
        return "Yellow", float(b / 255.0)
    if 85 < h <= 135 and s > 35:
        return "Blue", float(s / 255.0)
    if 40 < h <= 85 and s > 35:
        return "Green", float(s / 255.0)
    if I < 45 or v < 45:
        return "Black", float(1.0 - I / 100.0)
    if I < 85 and a > 128 and b > 128:
        return "Brown", float(1.0 - I / 150.0)
    if s < 25 and I > 150:
        return "White", float(I / 255.0)
    if s < 30 and 45 <= I <= 150:
        return "Gray", 0.5

    return "Bare_Body", 0.4


def train_and_benchmark_models(X_train_pca, y_train, X_val_pca, y_val):
    """Trains k-NN, RBF-SVM, and Random Forest on PCA features for both binary and multi-class tasks."""
    models = {
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=min(5, max(1, len(X_train_pca) // 2))),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=6, class_weight="balanced", random_state=42)
    }

    unique_train = np.unique(y_train)
    avg_mode = "binary" if len(unique_train) == 2 and set(unique_train).issubset({0, 1}) else "macro"

    metrics_summary = {}
    best_f1 = -1.0
    best_model_name = "Random Forest"

    for name, model in models.items():
        try:
            model.fit(X_train_pca, y_train)
            if X_val_pca is not None and len(X_val_pca) > 0 and y_val is not None and len(y_val) > 0:
                preds = model.predict(X_val_pca)
                acc = accuracy_score(y_val, preds)
                p, r, f1, _ = precision_recall_fscore_support(y_val, preds, average=avg_mode, zero_division=0)
            else:
                preds = model.predict(X_train_pca)
                acc = accuracy_score(y_train, preds)
                p, r, f1, _ = precision_recall_fscore_support(y_train, preds, average=avg_mode, zero_division=0)

            metrics_summary[name] = {"Accuracy": float(acc), "Precision": float(p), "Recall": float(r), "F1": float(f1)}
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
        except Exception:
            metrics_summary[name] = {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0}

    return models, metrics_summary, best_model_name


# ==============================================================================
# BATCH HARVESTING WITH IMAGE-LEVEL SPLIT
# ==============================================================================

def harvest_proposals_from_files(files, max_images=6, target_size=(64, 64)):
    """Harvests candidate crops, masks, and metadata across image files."""
    all_crops, all_masks, all_meta = [], [], []
    for f in files[:max_images]:
        try:
            raw_rgb = load_normalized_image(f)
            clean_rgb, _, _, _, _, _, _ = preprocess_image(raw_rgb)
            mask, _, _ = segment_foreground(clean_rgb)
            crops, masks, meta = extract_proposals(clean_rgb, mask, target_size=target_size)
            for m in meta:
                m["image_file"] = os.path.basename(f)
            all_crops.extend(crops)
            all_masks.extend(masks)
            all_meta.extend(meta)
        except Exception:
            continue
    return all_crops, all_masks, all_meta


def create_proposal_montage(crops, max_show=10, border_px=2):
    """Creates a 2D grid montage of 64x64 proposal crops."""
    if not crops:
        return None
    n_show = min(len(crops), max_show)
    selected = crops[:n_show]
    h, w = selected[0].shape[:2]

    n_cols = 5
    n_rows = (n_show + n_cols - 1) // n_cols

    montage_h = n_rows * (h + border_px) + border_px
    montage_w = n_cols * (w + border_px) + border_px
    montage = np.ones((montage_h, montage_w, 3), dtype=np.uint8) * 240

    for idx, c in enumerate(selected):
        r = idx // n_cols
        col = idx % n_cols
        y = border_px + r * (h + border_px)
        x = border_px + col * (w + border_px)
        montage[y:y + h, x:x + w] = c

    return montage


# ==============================================================================
# END-TO-END PIPELINE DRIVER (STAGES 1 TO 5)
# ==============================================================================

def run_pipeline(target_dataset="both", custom_image=None):
    """Executes the unified Machine Vision pipeline across Stages 1 to 5."""
    print("=" * 75)
    print("Machine Vision (UFMFRR-15-M) - Unified Pipeline (Stages 1 - 5)")
    print("Primary Contributor: Jakub Wrzyszcz")
    print("=" * 75)

    os.makedirs("output", exist_ok=True)
    coco_annotations = load_coco_annotations()

    if custom_image:
        if target_dataset in ["apple", "resistor"]:
            domain = target_dataset
        else:
            raw_temp = load_normalized_image(custom_image)
            lab_temp = cv2.cvtColor(raw_temp, cv2.COLOR_RGB2LAB)
            border_L = float(np.median(np.concatenate([
                lab_temp[0, :, 0], lab_temp[-1, :, 0], lab_temp[:, 0, 0], lab_temp[:, -1, 0]
            ])))
            domain = "resistor" if border_L > 150.0 else "apple"

        bg_files = resolve_dataset_files(domain)
        targets = [(domain, bg_files, custom_image)]
    elif target_dataset == "both":
        targets = [
            ("apple", resolve_dataset_files("apple"), None),
            ("resistor", resolve_dataset_files("resistor"), None),
        ]
    else:
        targets = [(target_dataset, resolve_dataset_files(target_dataset), None)]

    for tag, files, test_override in targets:
        print(f"\n{'='*30} DOMAIN: {tag.upper()} {'='*30}")
        if not files:
            print(f"[ERROR] No image files found for {tag}.")
            continue

        # Image-level Train / Validation Split (Prevents Patch-Level Data Leakage)
        if len(files) >= 4:
            train_files, val_files = train_test_split(files, test_size=0.30, random_state=42)
        else:
            train_files, val_files = files, files

        sample_file = test_override if test_override else (val_files[0] if val_files else train_files[0])
        raw_rgb = load_normalized_image(sample_file)
        print(f"Sample Image: {os.path.basename(sample_file)} (Resolution: {raw_rgb.shape})")

        # Stage 1: Preprocessing
        t0 = time.perf_counter()
        clean_rgb, h, s, v, i_chan, l_raw, l_eq = preprocess_image(raw_rgb)
        t_stage1 = (time.perf_counter() - t0) * 1000.0
        print(f"  [Stage 1] Preprocessing complete in {t_stage1:.2f} ms")

        # Stage 2: Foreground Segmentation & K-Means Benchmark
        t1 = time.perf_counter()
        mask, seg_info, contours = segment_foreground(clean_rgb)
        t_stage2 = (time.perf_counter() - t1) * 1000.0

        t_km0 = time.perf_counter()
        cluster_mask, _ = benchmark_clustering(clean_rgb, k=3)
        t_km = (time.perf_counter() - t_km0) * 1000.0
        print(f"  [Stage 2] {seg_info['method'].capitalize()} ({len(contours)} blobs): {t_stage2:.2f} ms | "
              f"K-Means (k=3): {t_km:.2f} ms (Speedup: {t_km/max(t_stage2, 0.001):.1f}x)")

        # Stage 3: Shape-Adaptive Proposal Generation
        t2 = time.perf_counter()
        sample_crops, sample_masks, sample_meta = extract_proposals(clean_rgb, mask)
        t_stage3 = (time.perf_counter() - t2) * 1000.0
        print(f"  [Stage 3] Extracted {len(sample_crops)} normalized 64x64 proposals in {t_stage3:.2f} ms")

        # Stage 4: 95D Feature Extraction & Train-Only PCA
        print(f"  [Stage 4] Harvesting proposals across {len(train_files[:5])} train images...")
        train_crops, train_masks, train_meta = harvest_proposals_from_files(train_files, max_images=5)
        val_crops, val_masks, val_meta = harvest_proposals_from_files(val_files, max_images=3)

        if len(train_crops) == 0:
            train_crops.extend(sample_crops)
            train_masks.extend(sample_masks)
            train_meta.extend(sample_meta)

        if len(train_crops) == 0:
            print(f"  [WARN] No proposals extracted for {tag}. Skipping Stages 4-5.")
            continue

        print(f"  [Stage 4] Extracting 95D features (Train: {len(train_crops)}, Val: {len(val_crops)})...")
        X_train = np.array([extract_95d_features(c, m) for c, m in zip(train_crops, train_masks)], dtype=np.float32)
        X_val = np.array([extract_95d_features(c, m) for c, m in zip(val_crops, val_masks)], dtype=np.float32) if val_crops else None

        scaler, pca, X_train_pca, X_val_pca = fit_scaler_and_pca(X_train, X_val, n_components=0.95)
        n_pca = pca.n_components_
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        compression = (1.0 - n_pca / 95.0) * 100.0
        print(f"  [Stage 4] PCA: 95D -> {n_pca}D ({compression:.1f}% reduction, {cum_var[-1]*100:.2f}% variance retained)")

        # Stage 5: Unified Machine Learning Classification, NMS & Counting Evaluation
        ml_summary = {}
        top_model_name = "Random Forest"
        counting_results = {}

        if tag == "apple":
            y_train = []
            for m in train_meta:
                fname = m.get("image_file", "")
                gt_boxes = coco_annotations.get(fname, [])
                p_box = m.get("bbox", [0, 0, 32, 32])
                max_iou = max([compute_box_iou(p_box, gb) for gb in gt_boxes]) if gt_boxes else 0.0
                y_train.append(1 if max_iou >= 0.25 else 0)
            y_train = np.array(y_train, dtype=int)

            y_val = None
            if val_meta:
                y_val_list = []
                for m in val_meta:
                    fname = m.get("image_file", "")
                    gt_boxes = coco_annotations.get(fname, [])
                    p_box = m.get("bbox", [0, 0, 32, 32])
                    max_iou = max([compute_box_iou(p_box, gb) for gb in gt_boxes]) if gt_boxes else 0.0
                    y_val_list.append(1 if max_iou >= 0.25 else 0)
                y_val = np.array(y_val_list, dtype=int)

            print(f"  [Stage 5] Ground-Truth Labels: {sum(y_train)} Apples, {len(y_train)-sum(y_train)} Background")
            models, ml_summary, top_model_name = train_and_benchmark_models(X_train_pca, y_train, X_val_pca, y_val)

            for m_name, met in ml_summary.items():
                print(f"    - {m_name:14s}: Acc={met['Accuracy']:.3f}, Prec={met['Precision']:.3f}, "
                      f"Rec={met['Recall']:.3f}, F1={met['F1']:.3f}")

            best_clf = models[top_model_name]

            # Sample Image Counting via 2D NMS
            pos_boxes, keep_nms = [], []
            if sample_crops:
                sample_feats = np.array([extract_95d_features(c, m) for c, m in zip(sample_crops, sample_masks)], dtype=np.float32)
                sample_pca = pca.transform(scaler.transform(sample_feats))
                sample_preds = best_clf.predict(sample_pca)
                if hasattr(best_clf, "predict_proba") and best_clf.predict_proba(sample_pca).shape[1] > 1:
                    sample_probs = best_clf.predict_proba(sample_pca)[:, 1]
                else:
                    sample_probs = sample_preds.astype(float)

                pos_indices = [i for i, p in enumerate(sample_preds) if p == 1]
                pos_boxes = [sample_meta[i]["bbox"] for i in pos_indices]
                pos_scores = [sample_probs[i] for i in pos_indices]
                keep_nms = apply_2d_nms(pos_boxes, pos_scores, iou_thresh=0.35)

            pred_apple_count = len(keep_nms)
            gt_apple_count = len(coco_annotations.get(os.path.basename(sample_file), []))
            err = abs(pred_apple_count - gt_apple_count)

            # Multi-Image Validation Set MAE and RMSE Evaluation
            y_true_counts, y_pred_counts = [], []
            for vf in val_files[:5]:
                v_rgb, _, _, _, _, _, _ = preprocess_image(load_normalized_image(vf))
                v_mask, _, _ = segment_foreground(v_rgb)
                v_crops, v_masks, v_meta = extract_proposals(v_rgb, v_mask)
                if v_crops:
                    v_feats = np.array([extract_95d_features(c, m) for c, m in zip(v_crops, v_masks)], dtype=np.float32)
                    v_pca = pca.transform(scaler.transform(v_feats))
                    v_preds = best_clf.predict(v_pca)
                    if hasattr(best_clf, "predict_proba") and best_clf.predict_proba(v_pca).shape[1] > 1:
                        v_probs = best_clf.predict_proba(v_pca)[:, 1]
                    else:
                        v_probs = v_preds.astype(float)
                    v_pos = [i for i, p in enumerate(v_preds) if p == 1]
                    v_keep = apply_2d_nms([v_meta[i]["bbox"] for i in v_pos], [v_probs[i] for i in v_pos], iou_thresh=0.35)
                    pred_c = len(v_keep)
                else:
                    pred_c = 0
                gt_c = len(coco_annotations.get(os.path.basename(vf), []))
                y_true_counts.append(gt_c)
                y_pred_counts.append(pred_c)

            mae = float(mean_absolute_error(y_true_counts, y_pred_counts)) if y_true_counts else float(err)
            rmse = float(np.sqrt(mean_squared_error(y_true_counts, y_pred_counts))) if y_true_counts else float(err)

            counting_results = {
                "Predicted Count": pred_apple_count,
                "Ground Truth Count": gt_apple_count,
                "Error": err,
                "MAE": mae,
                "RMSE": rmse,
                "Best Model": top_model_name
            }
            print(f"  [Stage 5] Sample Image Count: Pred={pred_apple_count} vs GT={gt_apple_count} (Abs Error: {err})")
            print(f"  [Stage 5] Validation Set Counting ({len(y_true_counts)} images): MAE={mae:.2f}, RMSE={rmse:.2f}")

        else:
            # Resistor Scenario: Train ML Classifiers on 95D PCA Features + 1D NMS Decoding
            y_train = np.array([
                classify_axial_slice(c, base_lab=m.get("base_lab"))[0]
                for c, m in zip(train_crops, train_meta)
            ])
            y_val = np.array([
                classify_axial_slice(c, base_lab=m.get("base_lab"))[0]
                for c, m in zip(val_crops, val_meta)
            ]) if val_crops else None

            models, ml_summary, top_model_name = train_and_benchmark_models(X_train_pca, y_train, X_val_pca, y_val)
            best_clf = models[top_model_name]

            print(f"  [Stage 5] Multi-Class Resistor Band ML Benchmark (Classes: {len(np.unique(y_train))}):")
            for m_name, met in ml_summary.items():
                print(f"    - {m_name:14s}: Acc={met['Accuracy']:.3f}, Macro-Prec={met['Precision']:.3f}, "
                      f"Macro-Rec={met['Recall']:.3f}, Macro-F1={met['F1']:.3f}")

            resistor_ids = sorted(list(set(m["resistor_id"] for m in sample_meta if "resistor_id" in m)))
            resistor_reports = []

            for r_id in resistor_ids:
                r_indices = [i for i, m in enumerate(sample_meta) if m.get("resistor_id") == r_id]
                r_slices_meta = [sample_meta[i] for i in r_indices]
                r_slice_crops = [sample_crops[i] for i in r_indices]
                r_slice_masks = [sample_masks[i] for i in r_indices]

                if not r_slice_crops:
                    continue

                # Predict slice classes using the trained ML classifier on PCA features
                r_feats = np.array([extract_95d_features(c, m) for c, m in zip(r_slice_crops, r_slice_masks)], dtype=np.float32)
                r_pca = pca.transform(scaler.transform(r_feats))
                ml_preds = list(best_clf.predict(r_pca))
                if hasattr(best_clf, "predict_proba"):
                    ml_confs = list(np.max(best_clf.predict_proba(r_pca), axis=1))
                else:
                    ml_confs = [1.0] * len(ml_preds)

                detected_bands = apply_1d_nms(r_slices_meta, ml_preds, ml_confs, bare_class="Bare_Body")
                band_names = [str(b[0]) for b in detected_bands]
                valid_bands, counts_dict = count_resistor_bands(band_names)

                report = {
                    "Resistor ID": r_id,
                    "BBox": r_slices_meta[0].get("bbox", [0, 0, 50, 50]),
                    "Bands": valid_bands,
                    "Band Counts": counts_dict,
                }
                resistor_reports.append(report)
                color_summary = ", ".join(f"{cnt} {col}" for col, cnt in counts_dict.items() if col != "Total Bands")
                print(f"    Resistor #{r_id} ({top_model_name}): Bands={valid_bands} -> "
                      f"Total: {counts_dict['Total Bands']} bands ({color_summary}) | Counts={counts_dict}")

            counting_results = {"Resistor Reports": resistor_reports, "Best Model": top_model_name}

        # ----------------------------------------------------------------------
        # Comprehensive 6-Panel Diagnostic Figure (Stages 1 to 5)
        # ----------------------------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))

        # (a) Stage 1 Preprocessed Image
        axes[0, 0].imshow(clean_rgb)
        axes[0, 0].set_title("(a) Stage 1: Preprocessed Image (CLAHE + Spatial Filter)")
        axes[0, 0].axis("off")

        # (b) Stage 2 Binary Foreground Mask
        axes[0, 1].imshow(mask, cmap="gray")
        axes[0, 1].set_title(f"(b) Stage 2: Binary Foreground Mask ({seg_info['method']})")
        axes[0, 1].axis("off")

        # (c) Stage 3 Standardized Proposals Montage
        axes[0, 2].axis("off")
        montage = create_proposal_montage(sample_crops, max_show=10)
        if montage is not None:
            axes[0, 2].imshow(montage)
            axes[0, 2].set_title(f"(c) Stage 3: Standardized Proposals ({min(len(sample_crops), 10)} shown)")
        else:
            axes[0, 2].text(0.5, 0.5, "No proposals extracted", ha="center", va="center")

        # (d) Stage 4 PCA Scree Plot
        axes[1, 0].plot(range(1, len(cum_var) + 1), cum_var, "o-", color="navy", linewidth=2)
        axes[1, 0].axhline(0.95, color="crimson", linestyle="--", label="95% Variance Threshold")
        axes[1, 0].axvline(n_pca, color="green", linestyle=":", label=f"Retained ({n_pca}D)")
        axes[1, 0].set_title(f"(d) Stage 4: PCA Scree Plot (95D -> {n_pca}D, {cum_var[-1]*100:.1f}%)")
        axes[1, 0].set_xlabel("Principal Components")
        axes[1, 0].set_ylabel("Cumulative Variance")
        axes[1, 0].set_ylim(0, 1.05)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # (e) Stage 5 Machine Learning Benchmark (Both Domains)
        if ml_summary:
            m_names = list(ml_summary.keys())
            f1s = [ml_summary[m]["F1"] for m in m_names]
            accs = [ml_summary[m]["Accuracy"] for m in m_names]
            x_ind = np.arange(len(m_names))
            axes[1, 1].bar(x_ind - 0.15, accs, 0.3, label="Accuracy", color="steelblue")
            axes[1, 1].bar(x_ind + 0.15, f1s, 0.3, label="F1-Score", color="darkorange")
            axes[1, 1].set_xticks(x_ind)
            axes[1, 1].set_xticklabels(m_names, rotation=15)
            axes[1, 1].set_ylim(0, 1.1)
            axes[1, 1].set_title(f"(e) Stage 5: ML Benchmark (Top: {top_model_name})")
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()

        # (f) Final Counting Result Overlay
        axes[1, 2].imshow(clean_rgb)
        if tag == "apple":
            axes[1, 2].set_title(
                f"(f) Stage 5 Count: Pred={counting_results.get('Predicted Count', 0)} | "
                f"GT={counting_results.get('Ground Truth Count', 0)} (MAE={counting_results.get('MAE', 0):.2f})"
            )
            for k in keep_nms:
                bx = pos_boxes[k]
                rect = plt.Rectangle((bx[0], bx[1]), bx[2], bx[3], fill=False, edgecolor="lime", linewidth=2)
                axes[1, 2].add_patch(rect)
        else:
            reps = counting_results.get("Resistor Reports", [])
            axes[1, 2].set_title(f"(f) Stage 5: Colour Band Counting ({len(reps)} Resistors)")
            for rep in reps:
                bx = rep["BBox"]
                rect = plt.Rectangle((bx[0], bx[1]), bx[2], bx[3], fill=False, edgecolor="cyan", linewidth=2)
                axes[1, 2].add_patch(rect)
                tot = rep["Band Counts"]["Total Bands"]
                label_txt = f"R#{rep['Resistor ID']}: {tot} bands"
                axes[1, 2].text(
                    bx[0], max(15, bx[1] - 6), label_txt,
                    color="yellow", fontsize=9, fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.6, pad=2)
                )
        axes[1, 2].axis("off")

        plt.tight_layout()
        out_name = "pipeline_custom_summary.png" if test_override else f"pipeline_{tag}_summary.png"
        out_fig = os.path.join("output", out_name)
        plt.savefig(out_fig, dpi=200, bbox_inches="tight")

        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                plt.show()
            else:
                plt.close()
        except Exception:
            plt.close()

        print(f"  -> Saved diagnostic figure: {out_fig}")

    print("\n" + "=" * 75)
    print("Unified Pipeline (Stages 1 - 5) successfully executed across all domains!")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Vision Unified Counting Pipeline (Stages 1 to 5)")
    parser.add_argument("--mode", type=str, default="both", choices=["both", "apple", "resistor"],
                        help="Dataset domain to process: 'both', 'apple', or 'resistor'")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an optional custom input image")
    args = parser.parse_args()

    run_pipeline(target_dataset=args.mode, custom_image=args.image)