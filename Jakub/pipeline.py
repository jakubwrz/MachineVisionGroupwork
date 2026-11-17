"""
Machine Vision (UFMFRR-15-M) - Group Project
Unified End-to-End Pipeline: Stages 1 to 4 (Apples & Resistors)

Primary Contributor: Jakub Wrzyszcz

Pipeline Architecture:
  Stage 1: Radiometric & Spatial Preprocessing (Units 2, 4, 5)
    - Decoupled Luminance CLAHE (CIE LAB L-channel) to normalize lighting without Hue shift.
    - Spatial Denoising: Median filtering (3x3 for apples, 5x5 for resistors to suppress specular streaks)
      followed by 3x3 Gaussian smoothing across RGB space.
    - True Unit 2 Intensity computation: I = (R + G + B) / 3.

  Stage 2: Candidate Foreground Segmentation (Units 3, 4)
    - Apples: Feature fusion of CIE LAB a* (redness) and HSV Saturation S, binarised via Otsu's thresholding.
      Benchmarked against cv2.kmeans (k=3) in 3D (S, a*, b*) feature space.
    - Resistors: Inverted Gaussian adaptive thresholding (cv2.adaptiveThreshold) with 5x5 morphological
      closing (cv2.MORPH_CLOSE) to bridge dark colour bands, keeping component bodies with area > 500 px.

  Stage 3: Morphology, Pose Alignment & Region Proposals (Unit 4)
    - Apples (Watershed Splitting): 3x3 opening, topological hole filling, and Euclidean Distance Transform
      (cv2.distanceTransform, L2 metric) to detect topological seeds for marker-controlled cv2.watershed,
      exporting standardized 64x64 candidate fruit crops.
    - Resistors (Pose & Slicing): cv2.minAreaRect and cv2.warpAffine horizontal alignment, stripping thin
      wire leads via vertical column thickness profile, and slicing bands via vertical Sobel filtering
      and 1D sliding windows, exporting standardized 64x64 band crops.

  Stage 4: Unified 95D Feature Extraction & PCA (Units 2, 4, 6, 7)
    - 32 Colour Descriptors: 20 statistical moments (mean, std, skewness, kurtosis across H, S, a*, b*, I)
      and 12 histogram bins (4 bins each for H, S, a*).
    - 59 Texture Descriptors: Rotation-invariant Uniform Local Binary Patterns (P=8, R=1, method='uniform').
    - 4 Invariant Shape Metrics: Circularity, Solidity, Aspect Ratio, Extent (excluding raw coordinates and area).
    - Dimensionality Reduction: StandardScaler and PCA(n_components=0.95) fit on the training split only.
"""

import os
import glob
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# ==============================================================================
# DATA RESOLUTION UTILITIES
# ==============================================================================

def resolve_dataset_files(mode="apple"):
    """Finds image file paths for either MinneApple or Resistor Dataset."""
    if mode == "apple":
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
            return files
    return []


def load_normalized_image(img_path, max_dim=1280):
    """Loads an image, standardizes max spatial dimension, and converts to RGB."""
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    h, w = bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = float(max_dim) / float(max(h, w))
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ==============================================================================
# STAGE 1: RADIOMETRIC & SPATIAL PREPROCESSING (UNITS 2, 4, 5)
# ==============================================================================

def compute_intensity(rgb_img):
    """Computes true Unit 2 Slide 25 Intensity: I = (R + G + B) / 3."""
    return np.mean(rgb_img.astype(np.float32), axis=2).astype(np.uint8)


def preprocess_image(rgb_img, mode="apple"):
    """
    Stage 1 Unified Preprocessing:
      1. Contrast Normalisation on Luminance (LAB L) via CLAHE.
      2. Spatial Denoising: Median filtering (3x3 for apples, 5x5 for resistors) + 3x3 Gaussian.
      3. Channel Extraction: Returns preprocessed RGB, H, S, V, Intensity I, raw L, and equalised L.
    """
    # 1. CLAHE on Luminance (L)
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l_raw, a_chan, b_chan = cv2.split(lab)

    clip = 2.5 if mode == "apple" else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_raw)
    eq_rgb = cv2.cvtColor(cv2.merge([l_eq, a_chan, b_chan]), cv2.COLOR_LAB2RGB)

    # 2. Spatial Denoising in RGB Space
    k_med = 3 if mode == "apple" else 5
    denoised_rgb = cv2.medianBlur(eq_rgb, k_med)
    clean_rgb = cv2.GaussianBlur(denoised_rgb, (3, 3), 0.8)

    # 3. Channels
    hsv = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    i_chan = compute_intensity(clean_rgb)

    return clean_rgb, h, s, v, i_chan, l_raw, l_eq


# ==============================================================================
# STAGE 2: FOREGROUND SEGMENTATION (UNITS 3, 4)
# ==============================================================================

def segment_apples_otsu(rgb_img, alpha=0.6):
    """Fuses CIE LAB a* and HSV Saturation S, followed by Otsu thresholding."""
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    a_chan = lab[:, :, 1].astype(np.float32)
    s_chan = hsv[:, :, 1].astype(np.float32)

    # Penalize green foliage chromaticity (neutral gray is 120-128 in uint8 LAB)
    a_diff = np.clip(a_chan - 120.0, 0.0, None)
    a_norm = cv2.normalize(a_diff, None, 0, 255, cv2.NORM_MINMAX)
    s_norm = cv2.normalize(s_chan, None, 0, 255, cv2.NORM_MINMAX)

    fused_map = (alpha * a_norm + (1.0 - alpha) * s_norm).astype(np.uint8)
    thresh_val, mask_otsu = cv2.threshold(fused_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask_otsu, fused_map, thresh_val


def benchmark_apple_kmeans(rgb_img, k=3):
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

    apple_idx = int(np.argmax(centers[:, 1]))
    mask_kmeans = (labels.reshape((h, w)) == apple_idx).astype(np.uint8) * 255
    return mask_kmeans, centers


def segment_resistors_adaptive(rgb_img, min_area=500):
    """Segments resistor bodies using adaptive Gaussian thresholding and 5x5 closing."""
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Inverted adaptive thresholding: dark resistors against bright bench -> foreground
    raw_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )

    # 5x5 Morphological closing to bridge across dark color bands
    kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(raw_thresh, cv2.MORPH_CLOSE, kernel_5x5, iterations=2)

    # Connected component filtering: retain blobs with area > min_area
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    body_mask = np.zeros_like(closed)
    resistor_blobs = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(body_mask, [c], -1, 255, thickness=cv2.FILLED)
            resistor_blobs.append(c)

    return body_mask, raw_thresh, resistor_blobs


def segment_foreground(rgb_img, mode="apple"):
    """Unified Stage 2 function returning candidate binary mask."""
    if mode == "apple":
        mask, fused, thresh = segment_apples_otsu(rgb_img)
        return mask, {"fused_map": fused, "thresh": thresh}
    else:
        mask, raw_thresh, blobs = segment_resistors_adaptive(rgb_img)
        return mask, {"raw_thresh": raw_thresh, "blobs": blobs}


# ==============================================================================
# STAGE 3: MORPHOLOGY, POSE ALIGNMENT & PROPOSALS (UNIT 4)
# ==============================================================================

def extract_apple_proposals(rgb_img, binary_mask, target_size=(64, 64)):
    """Watershed splitting via Euclidean Distance Transform and proposal cropping."""
    # 1. 3x3 Opening and Hole-Filling
    kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_3x3)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(opened)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    # 2. Euclidean Distance Transform (L2)
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    max_d = dist.max()
    sure_fg = np.zeros_like(filled)
    if max_d > 0:
        _, sure_fg = cv2.threshold(dist, 0.30 * max_d, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    sure_bg = cv2.dilate(filled, kernel_3x3, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(rgb_img.copy(), markers)

    # 3. Crop 64x64 Proposals
    h_img, w_img = rgb_img.shape[:2]
    crops, masks, metadata = [], [], []

    for label in np.unique(markers):
        if label < 2:
            continue
        inst_mask = (markers == label).astype(np.uint8) * 255
        area = int(np.sum(inst_mask > 0))
        if area < 50:
            continue

        M = cv2.moments(inst_mask)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(inst_mask)

        side = max(w, h, 32)
        half_side = int(side * 1.4) // 2
        x1, x2 = max(0, cx - half_side), min(w_img, cx + half_side)
        y1, y2 = max(0, cy - half_side), min(h_img, cy + half_side)

        patch = rgb_img[y1:y2, x1:x2]
        pmask = inst_mask[y1:y2, x1:x2]
        if patch.shape[0] < 4 or patch.shape[1] < 4:
            continue

        crops.append(cv2.resize(patch, target_size, interpolation=cv2.INTER_AREA))
        masks.append(cv2.resize(pmask, target_size, interpolation=cv2.INTER_NEAREST))
        metadata.append({"centroid": (cx, cy), "bbox": (x, y, w, h), "area": area})

    return crops, masks, metadata, {"dist_map": dist, "markers": markers}


def extract_resistor_proposals(rgb_img, body_mask, resistor_blobs, target_size=(64, 64)):
    """Horizontal alignment, wire lead stripping, and vertical Sobel band slicing."""
    h_img, w_img = rgb_img.shape[:2]
    crops, masks, metadata = [], [], []
    aligned_bodies = []

    for idx, c in enumerate(resistor_blobs):
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), angle = rect

        # Align longest axis horizontally
        if rw < rh:
            rot_angle = angle - 90.0 if angle >= 0 else angle + 90.0
            length, thickness = rh, rw
        else:
            rot_angle = angle
            length, thickness = rw, rh

        if length < 60 or thickness < 12:
            continue

        M = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
        rot_img = cv2.warpAffine(rgb_img, M, (w_img, h_img), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rot_mask = cv2.warpAffine(body_mask, M, (w_img, h_img), flags=cv2.INTER_NEAREST)

        x1 = int(max(0, cx - length / 2))
        x2 = int(min(w_img, cx + length / 2))
        y1 = int(max(0, cy - thickness))
        y2 = int(min(h_img, cy + thickness))

        crop_res = rot_img[y1:y2, x1:x2]
        crop_m = rot_mask[y1:y2, x1:x2]
        if crop_res.shape[1] < 40 or crop_res.shape[0] < 10:
            continue

        # Strip wire leads by vertical column thickness profile
        col_thick = np.sum(crop_m > 0, axis=0)
        max_thick = col_thick.max()
        if max_thick == 0:
            continue
        body_cols = np.where(col_thick > 0.35 * max_thick)[0]
        if len(body_cols) > 30:
            bx1, bx2 = body_cols[0], body_cols[-1]
            body_crop = crop_res[:, bx1:bx2]
            body_mask_crop = crop_m[:, bx1:bx2]
        else:
            body_crop = crop_res
            body_mask_crop = crop_m

        aligned_bodies.append(body_crop)

        # Slice bands via vertical Sobel on central 60%
        bh, bw = body_crop.shape[:2]
        if bw < 30 or bh < 10:
            continue
        center_strip = body_crop[int(bh * 0.2):int(bh * 0.8), :]
        gray_strip = cv2.cvtColor(center_strip, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray_strip, cv2.CV_64F, 1, 0, ksize=3)

        window_w = max(16, int(bw * 0.12))
        step = max(8, int(window_w * 0.8))

        for wx in range(0, bw - window_w, step):
            b_patch = body_crop[:, wx:wx + window_w]
            bm_patch = body_mask_crop[:, wx:wx + window_w]

            crops.append(cv2.resize(b_patch, target_size, interpolation=cv2.INTER_AREA))
            masks.append(cv2.resize(bm_patch, target_size, interpolation=cv2.INTER_NEAREST))
            metadata.append({
                "resistor_id": idx,
                "window_x": wx,
                "sobel_score": float(np.mean(np.abs(sobel_x[:, wx:wx + window_w]))),
            })

    return crops, masks, metadata, {"aligned_bodies": aligned_bodies}


def extract_proposals(rgb_img, mask, mode="apple", extra_data=None):
    """Unified Stage 3 entry point."""
    if mode == "apple":
        return extract_apple_proposals(rgb_img, mask)
    else:
        blobs = extra_data.get("blobs", []) if extra_data else []
        if not blobs:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blobs = [c for c in contours if cv2.contourArea(c) > 500]
        return extract_resistor_proposals(rgb_img, mask, blobs)


# ==============================================================================
# STAGE 4: UNIFIED 95D FEATURE EXTRACTION & PCA (UNITS 2, 4, 6, 7)
# ==============================================================================

def extract_95d_features(crop_rgb, crop_mask=None):
    """
    Extracts the unified 95D feature vector for any candidate crop:
      - 32 Colour Features: 20 statistical moments + 12 histogram bins across H, S, a*, b*, I.
      - 59 Texture Features: Uniform LBP (P=8, R=1, method='uniform').
      - 4 Shape Metrics: Circularity, Solidity, Aspect Ratio, Extent.
    """
    # 1. Colour Features (32D)
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    h_ch, s_ch, _ = cv2.split(hsv)
    _, a_ch, b_ch = cv2.split(lab)
    i_ch = compute_intensity(crop_rgb)

    channels = [h_ch, s_ch, a_ch, b_ch, i_ch]
    moments = []
    for ch in channels:
        vals = ch.ravel().astype(np.float32)
        moments.extend([
            float(np.mean(vals)),
            float(np.std(vals)),
            float(stats.skew(vals)),
            float(stats.kurtosis(vals)),
        ])

    h_hist, _ = np.histogram(h_ch.ravel(), bins=4, range=(0, 180), density=True)
    s_hist, _ = np.histogram(s_ch.ravel(), bins=4, range=(0, 256), density=True)
    a_hist, _ = np.histogram(a_ch.ravel(), bins=4, range=(0, 256), density=True)
    colour_feats = np.concatenate([moments, h_hist, s_hist, a_hist]).astype(np.float32)

    # 2. Texture Features (59D Uniform LBP)
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    lbp_map = local_binary_pattern(gray, P=8, R=1, method="uniform")
    texture_feats, _ = np.histogram(lbp_map.ravel(), bins=59, range=(0, 59), density=True)
    texture_feats = texture_feats.astype(np.float32)

    # 3. Shape Metrics (4D)
    if crop_mask is None:
        _, crop_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        perimeter = float(cv2.arcLength(c, True))
        circularity = (4.0 * np.pi * area) / ((perimeter**2) + 1e-6)
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
    """Fits StandardScaler and PCA on training split only to prevent data leakage."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=n_components, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_scaled)

    X_val_pca = None
    if X_val is not None and len(X_val) > 0:
        X_val_pca = pca.transform(scaler.transform(X_val))

    return scaler, pca, X_train_pca, X_val_pca


# ==============================================================================
# UNIFIED PIPELINE RUNNER & REPORT GENERATOR
# ==============================================================================

def harvest_dataset_proposals(files, mode="apple", max_images=5):
    """Gathers crops and masks from multiple dataset images for robust PCA fitting."""
    all_crops, all_masks = [], []
    for f in files[:max_images]:
        try:
            img = load_normalized_image(f)
            clean, _, _, _, _, _, _ = preprocess_image(img, mode=mode)
            mask, extra = segment_foreground(clean, mode=mode)
            crops, masks, _, _ = extract_proposals(clean, mask, mode=mode, extra_data=extra)
            all_crops.extend(crops)
            all_masks.extend(masks)
        except Exception as e:
            continue
    return all_crops, all_masks


def run_pipeline(mode="both"):
    """
    Executes the unified end-to-end Machine Vision pipeline (Stages 1 through 4)
    across Apples and Resistors, logging metrics and saving summary figures.
    """
    print("=" * 65)
    print("Machine Vision (UFMFRR-15-M) - Unified Pipeline (Stages 1 - 4)")
    print("Author: Jakub Wrzyszcz")
    print("=" * 65)

    os.makedirs("output", exist_ok=True)
    tasks = ["apple", "resistor"] if mode == "both" else [mode]

    for task in tasks:
        print(f"\n>>> EXECUTING PIPELINE FOR: {task.upper()} <<<")
        files = resolve_dataset_files(task)
        if not files:
            print(f"[ERROR] No dataset files found for {task}.")
            continue

        sample_file = files[0]
        raw_rgb = load_normalized_image(sample_file)
        print(f"Sample Image: {os.path.basename(sample_file)} (Shape: {raw_rgb.shape})")

        # Stage 1: Preprocessing
        t0 = time.perf_counter()
        clean_rgb, h, s, v, i_chan, l_raw, l_eq = preprocess_image(raw_rgb, mode=task)
        t_stage1 = (time.perf_counter() - t0) * 1000.0
        print(f"  [Stage 1] Preprocessing complete in {t_stage1:.2f} ms")

        # Stage 2: Segmentation
        t1 = time.perf_counter()
        mask, seg_extra = segment_foreground(clean_rgb, mode=task)
        t_stage2 = (time.perf_counter() - t1) * 1000.0
        if task == "apple":
            t_km0 = time.perf_counter()
            mask_km, _ = benchmark_apple_kmeans(clean_rgb, k=3)
            t_km = (time.perf_counter() - t_km0) * 1000.0
            print(f"  [Stage 2] Otsu: {t_stage2:.2f} ms | K-Means: {t_km:.2f} ms (Speedup: {t_km/t_stage2:.1f}x)")
        else:
            n_blobs = len(seg_extra.get("blobs", []))
            print(f"  [Stage 2] Adaptive Thresholding & 5x5 Closing: {t_stage2:.2f} ms ({n_blobs} bodies)")

        # Stage 3: Proposals
        t2 = time.perf_counter()
        sample_crops, sample_masks, meta, prop_extra = extract_proposals(
            clean_rgb, mask, mode=task, extra_data=seg_extra
        )
        t_stage3 = (time.perf_counter() - t2) * 1000.0
        print(f"  [Stage 3] Extracted {len(sample_crops)} normalized 64x64 proposals in {t_stage3:.2f} ms")

        # Stage 4: Feature Extraction & PCA Fitting on Batch
        print("  [Stage 4] Harvesting batch proposals for PCA fitting...")
        batch_crops, batch_masks = harvest_dataset_proposals(files, mode=task, max_images=5)
        print(f"  [Stage 4] Assembled {len(batch_crops)} proposals. Extracting 95D feature vectors...")

        X = np.array([extract_95d_features(c, m) for c, m in zip(batch_crops, batch_masks)], dtype=np.float32)
        X_train, X_val = train_test_split(X, test_size=0.20, random_state=42)
        scaler, pca, X_train_pca, X_val_pca = fit_scaler_and_pca(X_train, X_val, n_components=0.95)

        n_pca = pca.n_components_
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        compression = (1.0 - n_pca / 95.0) * 100.0
        print(f"  [Stage 4] PCA: 95D -> {n_pca}D ({compression:.1f}% reduction, {cum_var[-1]*100:.2f}% variance)")

        # 4-Panel Summary Diagnostic Visualisation
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (a) Stage 1
        axes[0, 0].imshow(clean_rgb)
        axes[0, 0].set_title(f"(a) Stage 1: Preprocessed {task.capitalize()} (CLAHE + Denoise)")
        axes[0, 0].axis("off")

        # (b) Stage 2
        axes[0, 1].imshow(mask, cmap="gray")
        axes[0, 1].set_title(f"(b) Stage 2: Binary Segmentation Mask ({task.capitalize()})")
        axes[0, 1].axis("off")

        # (c) Stage 3
        axes[1, 0].axis("off")
        n_show = min(len(sample_crops), 10)
        if n_show > 0:
            strip = np.hstack([sample_crops[i] for i in range(n_show)])
            axes[1, 0].imshow(strip)
            axes[1, 0].set_title(f"(c) Stage 3: Standardized 64x64 Proposals ({n_show} shown)")
        else:
            axes[1, 0].text(0.5, 0.5, "No proposals extracted", ha="center")

        # (d) Stage 4
        axes[1, 1].plot(range(1, len(cum_var) + 1), cum_var, "o-", color="navy", linewidth=2)
        axes[1, 1].axhline(0.95, color="crimson", linestyle="--", label="95% Variance Threshold")
        axes[1, 1].axvline(n_pca, color="green", linestyle=":", label=f"Retained ({n_pca}D)")
        axes[1, 1].set_title(f"(d) Stage 4 PCA Scree Plot: 95D -> {n_pca}D ({cum_var[-1]*100:.1f}%)")
        axes[1, 1].set_xlabel("Principal Components")
        axes[1, 1].set_ylabel("Cumulative Explained Variance")
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        out_fig = os.path.join("output", f"pipeline_{task}_summary.png")
        plt.savefig(out_fig, dpi=200, bbox_inches="tight")

        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                plt.show()
            else:
                plt.close()
        except Exception:
            plt.close()

        print(f"  -> Saved summary visualization: {out_fig}")

    print("\n" + "=" * 65)
    print("Unified Pipeline successfully executed across both tasks!")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline()
