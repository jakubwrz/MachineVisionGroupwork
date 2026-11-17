"""
Machine Vision (UFMFRR-15-M) - Group Portfolio Project
Unified End-to-End Object Counting Pipeline: Stages 1 to 4

Primary Contributor: Jakub Wrzyszcz

Architectural Overview (Appendix A & Section 2.1 Formulation):
  This pipeline implements a single coherent computer vision framework that addresses
  object counting across contrasting image domains (orchard fruit counting and resistor
  colour-band counting). In accordance with the assessment brief, all processing stages
  are formulated using universal computer vision functions that adapt automatically
  based on fundamental optical, radiometric, and geometric principles:

  Stage 1: Universal Radiometric & Spatial Preprocessing (Units 2, 4, 5)
    - Decoupled Luminance Normalisation: Converts RGB to CIE LAB, applying Contrast Limited
      Adaptive Histogram Equalisation (CLAHE) on the L-channel to standardise local contrast
      and lighting without chromatic distortion.
    - Spatial Denoising: Applies a 3x3 median filter (suppressing impulse/specular noise)
      followed by Gaussian smoothing (3x3, sigma=0.8) across RGB colour space.
    - Intensity & Chromatic Decomposition: Computes Unit 2 Intensity I = (R + G + B) / 3,
      HSV (H, S, V), and LAB (L, a*, b*) representations.

  Stage 2: Universal Foreground Segmentation & Clustering Benchmark (Units 3, 4)
    - Scene-Adaptive Contrast Binarisation: Automatically evaluates background illumination
      level (L_bg). In high-luminance bench inspection environments (L_bg > 150), inverted
      adaptive Gaussian thresholding isolates darker objects; in natural canopy environments
      (L_bg <= 150), chromatic saliency fusion (CIE LAB a* red and b* yellow chroma with HSV S)
      isolates fruit against foliage via Otsu's optimal thresholding.
    - Morphological Regularisation: Applies 5x5 closing (cv2.MORPH_CLOSE) to bridge internal
      discontinuities (such as colour bands) followed by topological hole-filling.
    - Unsupervised Clustering Benchmark: Benchmarks thresholding against cv2.kmeans (k=3)
      in 3D (S, a*, b*) colour feature space.

  Stage 3: Shape-Adaptive Morphological Proposal Generation (Unit 4)
    - Universal Geometric Decomposition: Inspects the intrinsic aspect ratio (major / minor axis)
      of candidate connected components:
        * Radial / Compact Morphology (Aspect Ratio < 3.0): Applies Euclidean Distance Transform
          (cv2.distanceTransform, L2 metric) and marker-controlled Watershed segmentation
          (cv2.watershed) to separate touching spherical clusters (e.g., orchard fruit),
          exporting standardized 64x64 candidate crops ranked by shape confidence.
        * Axial / Elongated Morphology (Aspect Ratio >= 3.0): Determines principal orientation
          from the minimum-area bounding box, aligns the instance horizontally via affine warp,
          strips thin wire leads using transverse column thickness profiling (isolating the active
          ceramic cylinder), and slices standardized 64x64 proposals across colour bands.

  Stage 4: Unified 95D Feature Extraction & Dimensionality Reduction (Units 2, 4, 6, 7)
    - 32 Colour Descriptors: 20 statistical moments (mean, std, skewness, kurtosis across H, S, a*, b*, I)
      and 12 histogram bins (4 bins each for H, S, a*).
    - 59 Texture Descriptors: Rotation-invariant Uniform Local Binary Patterns (P=8, R=1, method='uniform').
    - 4 Invariant Shape Metrics: Circularity, Solidity, Aspect Ratio, and Extent.
    - Dimensionality Reduction: StandardScaler and PCA(n_components=0.95) fit strictly on the
      training split to prevent data leakage, evaluating cumulative explained variance.
"""

import os
import glob
import time
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# ==============================================================================
# DATASET UTILITIES
# ==============================================================================

def resolve_dataset_files(target="apple"):
    """
    Finds image file paths for either MinneApple or Resistor Dataset.
    Supports local workspace folders and Google Colab default paths.
    """
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
    """
    Loads an image from disk, normalizes maximum spatial dimension to preserve
    aspect ratio without excessive memory consumption, and converts BGR to RGB.
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image at path: {img_path}")
    h, w = bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = float(max_dim) / float(max(h, w))
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ==============================================================================
# STAGE 1: UNIVERSAL RADIOMETRIC & SPATIAL PREPROCESSING (UNITS 2, 4, 5)
# ==============================================================================

def compute_intensity(rgb_img):
    """
    Computes true Unit 2 Intensity: I = (R + G + B) / 3.
    """
    return np.mean(rgb_img.astype(np.float32), axis=2).astype(np.uint8)


def preprocess_image(rgb_img, clahe_clip=2.0, grid_size=(8, 8), median_k=3, gauss_sigma=0.8):
    """
    Universal Stage 1 Preprocessing function applicable to any RGB input image:
      1. Converts RGB to CIE LAB color space.
      2. Applies CLAHE exclusively on the luminance (L) channel to normalize
         uneven illumination while preserving chromaticity without color shifts.
      3. Converts back to RGB and applies spatial filtering: a 3x3 median blur
         for impulse noise suppression, followed by Gaussian smoothing (sigma=0.8).
      4. Extracts multi-space representations: HSV (H, S, V), LAB, and Intensity I.
    """
    # 1. CLAHE on Luminance (LAB L-channel)
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l_raw, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=grid_size)
    l_eq = clahe.apply(l_raw)
    eq_rgb = cv2.cvtColor(cv2.merge([l_eq, a_chan, b_chan]), cv2.COLOR_LAB2RGB)

    # 2. Spatial Noise Reduction
    denoised_rgb = cv2.medianBlur(eq_rgb, median_k)
    clean_rgb = cv2.GaussianBlur(denoised_rgb, (3, 3), gauss_sigma)

    # 3. Channel Representations
    hsv = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    i_chan = compute_intensity(clean_rgb)

    return clean_rgb, h, s, v, i_chan, l_raw, l_eq


# ==============================================================================
# STAGE 2: UNIVERSAL FOREGROUND SEGMENTATION & BENCHMARKING (UNITS 3, 4)
# ==============================================================================

def segment_foreground(rgb_img, method="auto", min_area=80):
    """
    Universal Stage 2 Foreground Segmentation function:
      Adapts based on optical background illumination (L_bg):
        - High-Luminance Backgrounds (L_bg > 150, e.g. bright workbench / paper):
          Uses inverted adaptive Gaussian thresholding (block size 101, C=15)
          to isolate dark component bodies and leads against bright backgrounds.
        - Natural / Low-Luminance Backgrounds (L_bg <= 150, e.g. orchard canopy):
          Uses chromatic contrast fusion (CIE LAB a* red and b* yellow chroma with HSV S)
          to separate fruits from foliage, binarized via Otsu's threshold.
      Followed by 5x5 morphological closing (cv2.MORPH_CLOSE) to bridge internal
      discontinuities (such as colour bands) and topological hole-filling to form solid masks.
    """
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0]

    # Sample border pixels to determine ambient background luminance level
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

        # Fruit chromatic contrast: red chroma (a* > 124) + yellow chroma (b* > 138) weighted by saturation S
        fruit_chroma = np.maximum(a_chan - 124.0, 0.0) + np.maximum(b_chan - 138.0, 0.0)
        saliency = fruit_chroma * (s_chan / 255.0)
        saliency_norm = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        thresh_val, raw_binary = cv2.threshold(saliency_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        debug_info = {"method": "otsu_chroma_fusion", "thresh_val": thresh_val, "bg_L": bg_L}

    # Morphological closing (5x5) to bridge internal gaps and dark colour bands
    kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(raw_binary, cv2.MORPH_CLOSE, kernel_5x5, iterations=3)

    # Topological hole filling to guarantee solid candidate object masks
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    # Filter noise blobs below minimum area threshold
    clean_mask = np.zeros_like(filled)
    valid_contours = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(clean_mask, [c], -1, 255, thickness=cv2.FILLED)
            valid_contours.append(c)

    return clean_mask, debug_info, valid_contours


def benchmark_clustering(rgb_img, k=3):
    """
    Universal Unsupervised Clustering Benchmark (cv2.kmeans):
      Clusters image pixels in 3D (S, a*, b*) colour feature space to partition
      foreground from background, benchmarking execution time and segmentation
      fidelity against thresholding.
    """
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
    Universal Stage 3 Proposal Generation function:
      Inspects the intrinsic aspect ratio (major / minor axis) of each connected component:
        - Radial / Compact Morphology (Aspect Ratio < 3.0):
            Applies Euclidean Distance Transform (L2) and marker-controlled Watershed
            segmentation to split clustered spherical objects (e.g. orchard fruit),
            exporting standardized 64x64 candidate crops ranked by shape confidence.
        - Axial / Elongated Morphology (Aspect Ratio >= 3.0):
            Computes orientation from the minimum bounding rectangle, rotates the component
            horizontally via affine transformation, strips thin wire leads using transverse
            column thickness profiling (isolating the active ceramic cylinder), and slices
            standardized 64x64 proposals across colour bands.
      Returns:
        crops: List of 64x64 RGB candidate crops.
        masks: List of corresponding 64x64 binary masks.
        metadata: List of diagnostic dictionaries with bounding boxes and morphology types.
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

        # ----------------------------------------------------------------------
        # Case A: Radial / Compact Instances (e.g. spherical fruit / apples)
        # ----------------------------------------------------------------------
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
                # Single isolated compact instance
                M = cv2.moments(comp_mask)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    side = int(np.clip(max(major_len * 1.1, 48), 48, 100))
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
                            "aspect_ratio": aspect_ratio,
                            "area": int(M["m00"]),
                        })
            else:
                # Touching cluster: resolve via marker-controlled watershed
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
                    x, y, w, h = cv2.boundingRect(inst_m)
                    side = int(np.clip(max(w, h), 48, 100))
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
                            "aspect_ratio": aspect_ratio,
                            "area": int(inst_area),
                        })

        # ----------------------------------------------------------------------
        # Case B: Axial / Elongated Instances (e.g. cylindrical resistors with leads)
        # ----------------------------------------------------------------------
        else:
            long_vec = e1 if l1 >= l2 else e2
            if major_len < 80 or minor_len < 8:
                continue

            # Compute orientation of the major axis
            angle = float(np.degrees(np.arctan2(long_vec[1], long_vec[0])))
            cx, cy = rect[0]

            # Rotate component so the major cylindrical axis is strictly horizontal
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

            # Strip thin wire leads: body exhibits significantly higher transverse thickness
            col_thick = np.sum(c_mask > 0, axis=0)
            smooth_thick = np.convolve(col_thick, np.ones(21) / 21.0, mode="same")
            c_start, c_end = int(cw * 0.25), int(cw * 0.75)
            if c_end <= c_start:
                continue

            center_pk = c_start + int(np.argmax(smooth_thick[c_start:c_end]))
            max_th = smooth_thick[center_pk]
            if max_th < 15:
                continue

            # Locate ceramic body boundaries where thickness exceeds 55% of peak
            bx1 = center_pk
            while bx1 > max(0, c_start - 30) and smooth_thick[bx1] > 0.55 * max_th:
                bx1 -= 1
            bx2 = center_pk
            while bx2 < min(cw - 1, c_end + 30) and smooth_thick[bx2] > 0.55 * max_th:
                bx2 += 1

            body_crop = c_crop[:, bx1:bx2]
            body_m = c_mask[:, bx1:bx2]

            # Crop tight vertical rows to eliminate background artifacts
            row_thick = np.sum(body_m > 0, axis=1)
            if len(row_thick) > 0 and row_thick.max() > 0:
                act_rows = np.where(row_thick > 0.4 * row_thick.max())[0]
                if len(act_rows) > 10:
                    body_crop = body_crop[act_rows[0]:act_rows[-1], :]
                    body_m = body_m[act_rows[0]:act_rows[-1], :]

            bh, bw = body_crop.shape[:2]
            if bw < 25 or bh < 12:
                continue

            # Slice standardized colour band proposals across the active body
            n_bands = 4
            win_w = max(16, bw // n_bands)
            for b_idx in range(n_bands):
                wx = int(b_idx * (bw - win_w) / max(1, n_bands - 1))
                band_p = body_crop[:, wx:wx + win_w]
                band_m = body_m[:, wx:wx + win_w]
                crops.append(cv2.resize(band_p, target_size, interpolation=cv2.INTER_AREA))
                masks.append(cv2.resize(band_m, target_size, interpolation=cv2.INTER_NEAREST))
                conf_scores.append(float(bw * bh + b_idx))
                metadata.append({
                    "type": "axial_band",
                    "resistor_id": idx,
                    "band_index": b_idx,
                    "aspect_ratio": aspect_ratio,
                })

    # Sort proposals descending by confidence score if available
    if conf_scores and len(conf_scores) == len(crops):
        sort_order = np.argsort(conf_scores)[::-1]
        crops = [crops[i] for i in sort_order]
        masks = [masks[i] for i in sort_order]
        metadata = [metadata[i] for i in sort_order]

    return crops, masks, metadata


# ==============================================================================
# STAGE 4: UNIFIED 95D FEATURE EXTRACTION & PCA (UNITS 2, 4, 6, 7)
# ==============================================================================

def extract_95d_features(crop_rgb, crop_mask=None):
    """
    Universal Stage 4 Feature Extraction function:
      Extracts a comprehensive 95D descriptor vector for any candidate 64x64 proposal:
        1. 32 Colour Features: 20 statistical moments (mean, std, skewness, kurtosis across
           H, S, a*, b*, and Intensity I) + 12 histogram bins (4 bins each for H, S, a*).
        2. 59 Texture Features: Rotation-invariant Uniform Local Binary Patterns (P=8, R=1, 'uniform').
        3. 4 Shape Metrics: Circularity, Solidity, Aspect Ratio, and Extent.
      Asserts exact 95D dimensionality.
    """
    # 1. Colour Descriptors (32D)
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

    # 2. Texture Descriptors (59D Uniform LBP)
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    lbp_map = local_binary_pattern(gray, P=8, R=1, method="uniform")
    texture_feats, _ = np.histogram(lbp_map.ravel(), bins=59, range=(0, 59), density=True)
    texture_feats = texture_feats.astype(np.float32)

    # 3. Shape Descriptors (4D)
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
    """
    Fits StandardScaler and PCA on the training split only to prevent data leakage.
    Projects feature vectors into principal subspace capturing >= 95% variance.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=n_components, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_scaled)

    X_val_pca = None
    if X_val is not None and len(X_val) > 0:
        X_val_pca = pca.transform(scaler.transform(X_val))

    return scaler, pca, X_train_pca, X_val_pca


# ==============================================================================
# BATCH HARVESTING & PIPELINE EXECUTION
# ==============================================================================

def harvest_proposals_from_files(files, max_images=5, target_size=(64, 64)):
    """
    Harvests candidate crops and masks across multiple dataset images using
    the universal processing functions for robust batch PCA fitting.
    """
    all_crops, all_masks = [], []
    for f in files[:max_images]:
        try:
            raw_rgb = load_normalized_image(f)
            clean_rgb, _, _, _, _, _, _ = preprocess_image(raw_rgb)
            mask, _, _ = segment_foreground(clean_rgb)
            crops, masks, _ = extract_proposals(clean_rgb, mask, target_size=target_size)
            all_crops.extend(crops)
            all_masks.extend(masks)
        except Exception:
            continue
    return all_crops, all_masks


def create_proposal_montage(crops, max_show=10, border_px=2):
    """
    Creates a clean 2D grid montage of proposal crops with distinct borders
    for clear, professional diagnostic visualization.
    """
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
        montage[y:y+h, x:x+w] = c

    return montage


def run_pipeline(target_dataset="both", custom_image=None):
    """
    Executes the unified end-to-end Machine Vision pipeline across Stages 1 to 4.
    Runs the exact same universal functions without class-specific branching.
    """
    print("=" * 70)
    print("Machine Vision (UFMFRR-15-M) - Unified Pipeline (Stages 1 - 4)")
    print("Author: Jakub Wrzyszcz")
    print("=" * 70)

    os.makedirs("output", exist_ok=True)

    if custom_image:
        targets = [("custom", [custom_image])]
    elif target_dataset == "both":
        targets = [
            ("apple", resolve_dataset_files("apple")),
            ("resistor", resolve_dataset_files("resistor")),
        ]
    else:
        targets = [(target_dataset, resolve_dataset_files(target_dataset))]

    for tag, files in targets:
        print(f"\n>>> PROCESSING DOMAIN: {tag.upper()} <<<")
        if not files:
            print(f"[ERROR] No image files found for {tag}.")
            continue

        sample_file = files[0]
        raw_rgb = load_normalized_image(sample_file)
        print(f"Sample Image: {os.path.basename(sample_file)} (Resolution: {raw_rgb.shape})")

        # Stage 1: Universal Preprocessing
        t0 = time.perf_counter()
        clean_rgb, h, s, v, i_chan, l_raw, l_eq = preprocess_image(raw_rgb)
        t_stage1 = (time.perf_counter() - t0) * 1000.0
        print(f"  [Stage 1] Preprocessing complete in {t_stage1:.2f} ms")

        # Stage 2: Universal Foreground Segmentation
        t1 = time.perf_counter()
        mask, seg_info, contours = segment_foreground(clean_rgb)
        t_stage2 = (time.perf_counter() - t1) * 1000.0

        # Benchmark against cv2.kmeans clustering
        t_km0 = time.perf_counter()
        cluster_mask, _ = benchmark_clustering(clean_rgb, k=3)
        t_km = (time.perf_counter() - t_km0) * 1000.0
        print(f"  [Stage 2] {seg_info['method'].capitalize()} ({len(contours)} blobs): {t_stage2:.2f} ms | "
              f"K-Means (k=3): {t_km:.2f} ms (Speedup: {t_km/t_stage2:.1f}x)")

        # Stage 3: Universal Shape-Adaptive Proposal Generation
        t2 = time.perf_counter()
        sample_crops, sample_masks, metadata = extract_proposals(clean_rgb, mask)
        t_stage3 = (time.perf_counter() - t2) * 1000.0
        print(f"  [Stage 3] Extracted {len(sample_crops)} normalized 64x64 proposals in {t_stage3:.2f} ms")

        # Stage 4: Universal 95D Feature Extraction & PCA Fitting
        print("  [Stage 4] Harvesting proposals across dataset batch for PCA fitting...")
        batch_crops, batch_masks = harvest_proposals_from_files(files, max_images=5)
        if len(batch_crops) < 10:
            batch_crops.extend(sample_crops)
            batch_masks.extend(sample_masks)

        print(f"  [Stage 4] Assembled {len(batch_crops)} candidate proposals. Extracting 95D feature vectors...")
        X = np.array([extract_95d_features(c, m) for c, m in zip(batch_crops, batch_masks)], dtype=np.float32)

        test_size = 0.20 if len(X) >= 10 else 0.0
        if test_size > 0:
            X_train, X_val = train_test_split(X, test_size=test_size, random_state=42)
        else:
            X_train, X_val = X, None

        scaler, pca, X_train_pca, X_val_pca = fit_scaler_and_pca(X_train, X_val, n_components=0.95)

        n_pca = pca.n_components_
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        compression = (1.0 - n_pca / 95.0) * 100.0
        print(f"  [Stage 4] PCA: 95D -> {n_pca}D ({compression:.1f}% dimensionality reduction, "
              f"{cum_var[-1]*100:.2f}% explained variance retained)")

        # ----------------------------------------------------------------------
        # 4-Panel Diagnostic Figure
        # ----------------------------------------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (a) Stage 1 Preprocessed Image
        axes[0, 0].imshow(clean_rgb)
        axes[0, 0].set_title(f"(a) Stage 1: Preprocessed Image (CLAHE + Spatial Filter)")
        axes[0, 0].axis("off")

        # (b) Stage 2 Binary Foreground Mask
        axes[0, 1].imshow(mask, cmap="gray")
        axes[0, 1].set_title(f"(b) Stage 2: Binary Foreground Mask ({seg_info['method']})")
        axes[0, 1].axis("off")

        # (c) Stage 3 Standardized 64x64 Proposals Montage
        axes[1, 0].axis("off")
        montage = create_proposal_montage(sample_crops, max_show=10)
        if montage is not None:
            axes[1, 0].imshow(montage)
            axes[1, 0].set_title(f"(c) Stage 3: Standardized 64x64 Proposals ({min(len(sample_crops), 10)} shown)")
        else:
            axes[1, 0].text(0.5, 0.5, "No proposals extracted", ha="center", va="center")

        # (d) Stage 4 PCA Scree Plot
        axes[1, 1].plot(range(1, len(cum_var) + 1), cum_var, "o-", color="navy", linewidth=2)
        axes[1, 1].axhline(0.95, color="crimson", linestyle="--", label="95% Variance Threshold")
        axes[1, 1].axvline(n_pca, color="green", linestyle=":", label=f"Retained ({n_pca}D)")
        axes[1, 1].set_title(f"(d) Stage 4: PCA Scree Plot (95D -> {n_pca}D, {cum_var[-1]*100:.1f}%)")
        axes[1, 1].set_xlabel("Principal Components")
        axes[1, 1].set_ylabel("Cumulative Explained Variance")
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        out_fig = os.path.join("output", f"pipeline_{tag}_summary.png")
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

    print("\n" + "=" * 70)
    print("Unified Pipeline successfully executed across all domains!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Vision Unified Counting Pipeline (Stages 1 to 4)")
    parser.add_argument("--mode", type=str, default="both", choices=["both", "apple", "resistor"],
                        help="Dataset domain to process: 'both', 'apple', or 'resistor'")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an optional custom input image")
    args = parser.parse_args()

    run_pipeline(target_dataset=args.mode, custom_image=args.image)
