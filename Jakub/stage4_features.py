"""
Machine Vision (UFMFRR-15-M) - Group Project
Stage 4: Feature Extraction & Scaling (Apples & Resistors: 95D Representation & PCA)

Primary Contributor: Jakub Wrzyszcz

Unified Pipeline Methodology:
  1. 95D Composite Feature Vector (Identical & Universal Across Both Tasks):
     - 32 Colour Descriptors (Units 2 & 4):
       * 20 Statistical Moments: Mean, Standard Deviation, Skewness, Kurtosis
         across 5 perceptual & physical channels: H, S, a*, b*, and true Unit 2 Intensity I = (R+G+B)/3.
       * 12 Colour Histogram Bins: 4 bins each for Hue (H), Saturation (S), and CIE LAB a*.
     - 59 Uniform LBP Texture Bins (Unit 6):
       * Rotation-invariant Uniform Local Binary Patterns (P=8, R=1, method='uniform').
       * 58 uniform bit transitions + 1 non-uniform pattern bucket = 59 bins.
       * In Task 1, separates smooth apple skin from rough bark/leaves;
         in Task 2, captures texture difference between metallic (Gold/Silver) and matte bands.
     - 4 Invariant Shape Metrics (Unit 4):
       * Circularity: 4 * pi * Area / Perimeter^2 (compactness).
       * Solidity: Area / ConvexHull Area (surface roughness / convexity).
       * Aspect Ratio: BoundingBox Width / Height.
       * Extent: Area / BoundingBox Area.
       * Excludes raw centroid (x, y) coordinates and unnormalized pixel area to guarantee
         scale and spatial translation invariance.
     * Total Dimensions: 32 + 59 + 4 = 95D.

  2. Statistical Normalisation & Dimensionality Reduction (Unit 7):
     - StandardScaler: Standardizes features to zero mean and unit variance.
     - Principal Component Analysis (PCA): Retains 95% of total explained variance.
     - CRITICALLY FIT ON TRAINING SPLIT ONLY to prevent data leakage.
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Ensure local imports work across environments
_curr_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
import sys
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

try:
    from stage2_segmentation import segment_apples_otsu, segment_resistors_adaptive
    from stage3_morphology_proposals import (
        apply_apple_morphology,
        split_apple_clusters_watershed,
        extract_apple_crops,
        align_and_slice_resistors,
    )
except ImportError:
    from Jakub.stage2_segmentation import segment_apples_otsu, segment_resistors_adaptive
    from Jakub.stage3_morphology_proposals import (
        apply_apple_morphology,
        split_apple_clusters_watershed,
        extract_apple_crops,
        align_and_slice_resistors,
    )


# ==============================================================================
# 95D FEATURE EXTRACTION (SHARED & IDENTICAL ACROSS TASKS)
# ==============================================================================

def extract_shape_metrics(crop_mask):
    """
    Computes 4 scale- and translation-invariant shape descriptors.
    Excludes raw centroid (x, y) and unnormalized pixel area.
    """
    if crop_mask is None or np.sum(crop_mask > 0) == 0:
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    contours, _ = cv2.findContours(
        crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perimeter = float(cv2.arcLength(c, True))

    # 1. Circularity: 4 * pi * Area / Perimeter^2
    circularity = (4.0 * np.pi * area) / ((perimeter**2) + 1e-6)

    # 2. Solidity: Area / Convex Hull Area
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / (hull_area + 1e-6)

    # 3. Aspect Ratio: Width / Height
    _, _, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / float(h + 1e-6)

    # 4. Extent: Area / Bounding Box Area
    extent = area / float(w * h + 1e-6)

    return np.array([circularity, solidity, aspect_ratio, extent], dtype=np.float32)


def extract_ulbp_texture(crop_rgb, P=8, R=1):
    """
    Extracts 59 Uniform Local Binary Pattern (uLBP) texture bins (P=8, R=1, uniform).
    58 uniform transitions + 1 non-uniform pattern bucket = 59 total bins.
    """
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    lbp_img = local_binary_pattern(gray, P=P, R=R, method="uniform")
    lbp_hist, _ = np.histogram(lbp_img.ravel(), bins=59, range=(0, 59), density=True)
    return lbp_hist.astype(np.float32), lbp_img


def extract_colour_descriptors(crop_rgb):
    """
    Extracts 32 colour features across 5 channels (H, S, a*, b*, I):
      - 20 Statistical Moments: Mean, Std, Skewness, Kurtosis for each channel.
      - 12 Colour Histogram Bins: 4 bins each for H (0-180), S (0-256), and a* (0-256).
    """
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)

    h_chan, s_chan, _ = cv2.split(hsv)
    _, a_chan, b_chan = cv2.split(lab)
    i_chan = np.mean(crop_rgb.astype(np.float32), axis=2).astype(np.uint8)

    channels = [h_chan, s_chan, a_chan, b_chan, i_chan]

    # 20 moments (4 per channel across 5 channels)
    moments = []
    for ch in channels:
        vals = ch.ravel().astype(np.float32)
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        skew = float(stats.skew(vals))
        kurt = float(stats.kurtosis(vals))
        moments.extend([mu, sigma, skew, kurt])

    # 12 histogram bins (4 bins each for H, S, a*)
    h_hist, _ = np.histogram(h_chan.ravel(), bins=4, range=(0, 180), density=True)
    s_hist, _ = np.histogram(s_chan.ravel(), bins=4, range=(0, 256), density=True)
    a_hist, _ = np.histogram(a_chan.ravel(), bins=4, range=(0, 256), density=True)

    colour_feats = np.concatenate([moments, h_hist, s_hist, a_hist]).astype(np.float32)
    return colour_feats


def extract_95d_features(crop_rgb, crop_mask=None):
    """
    Unified 95D feature extraction function applicable to any 64x64 candidate crop
    (Apple fruit patch or Resistor color-band slice).
    """
    colour_feats = extract_colour_descriptors(crop_rgb)
    texture_feats, _ = extract_ulbp_texture(crop_rgb)

    if crop_mask is None:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        _, crop_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    shape_feats = extract_shape_metrics(crop_mask)

    feat_vector = np.concatenate([colour_feats, texture_feats, shape_feats])
    assert len(feat_vector) == 95, f"Expected 95D vector, but got {len(feat_vector)}D"
    return feat_vector


def fit_scaler_and_pca(X_train, X_val=None, n_components=0.95):
    """
    Fits StandardScaler and PCA on training split only to prevent data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=n_components, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_scaled)

    X_val_pca = None
    if X_val is not None and len(X_val) > 0:
        X_val_scaled = scaler.transform(X_val)
        X_val_pca = pca.transform(X_val_scaled)

    return scaler, pca, X_train_pca, X_val_pca


# ==============================================================================
# PIPELINE HARVESTERS & DIAGNOSTIC PLOTTING
# ==============================================================================

def harvest_apple_proposals(img_paths, max_images=5):
    """Harvests candidate 64x64 apple proposals from training images."""
    crops = []
    masks = []
    for p in img_paths[:max_images]:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raw_mask, _, _ = segment_apples_otsu(rgb)
        opened, filled = apply_apple_morphology(raw_mask)
        _, markers, _ = split_apple_clusters_watershed(rgb, filled)
        c, m, _ = extract_apple_crops(rgb, markers, min_area=40)
        crops.extend(c)
        masks.extend(m)
    return crops, masks


def harvest_resistor_proposals(img_paths, max_images=5):
    """Harvests candidate 64x64 band crops from real resistor dataset images."""
    crops = []
    masks = []
    for p in img_paths[:max_images]:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        if max(h, w) > 1280:
            scale = 1280.0 / max(h, w)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        body_mask, _, blobs = segment_resistors_adaptive(rgb, min_area=500)
        c, m, _, _ = align_and_slice_resistors(rgb, blobs, body_mask)
        crops.extend(c)
        masks.extend(m)
    return crops, masks


def evaluate_and_plot_task(task_name, crops, masks, out_filename):
    """Evaluates 95D features + PCA and saves 4-panel diagnostic figure."""
    print(f"\n[{task_name.upper()}] Assembling 95D features from {len(crops)} crops...")
    feature_matrix = [extract_95d_features(c, m) for c, m in zip(crops, masks)]
    X = np.array(feature_matrix, dtype=np.float32)

    X_train, X_val = train_test_split(X, test_size=0.20, random_state=42)
    scaler, pca, X_train_pca, X_val_pca = fit_scaler_and_pca(X_train, X_val, n_components=0.95)

    n_orig = X.shape[1]
    n_pca = pca.n_components_
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    print("-" * 55)
    print(f"{task_name.upper()} PCA REPORT:")
    print(f"  - Original Dimensions:     {n_orig}D")
    print(f"  - Retained PCA Components: {n_pca}D (for >=95% variance)")
    print(f"  - Dimension Reduction:     {(1.0 - n_pca / n_orig) * 100:.1f}%")
    print(f"  - Total Variance Retained: {cum_var[-1] * 100:.2f}%")
    print("-" * 55)

    # 4-Panel Report Diagnostic Figure
    sample_crop = crops[0]
    sample_lbp_hist, sample_lbp_map = extract_ulbp_texture(sample_crop)
    sample_col_feats = extract_colour_descriptors(sample_crop)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Crop + LBP Map
    combined_vis = np.hstack([
        sample_crop,
        cv2.cvtColor((sample_lbp_map * (255.0 / 58.0)).astype(np.uint8), cv2.COLOR_GRAY2RGB),
    ])
    axes[0, 0].imshow(combined_vis)
    axes[0, 0].set_title(f"(a) {task_name.capitalize()} Crop & Uniform LBP Map")
    axes[0, 0].axis("off")

    # (b) 32D Colour Features
    axes[0, 1].bar(range(32), sample_col_feats, color="darkorange", alpha=0.75, width=0.8)
    axes[0, 1].axvline(19.5, color="red", linestyle="--", label="Moments (20) | Histograms (12)")
    axes[0, 1].set_title(f"(b) 32D Colour Features (H, S, a*, b*, I)")
    axes[0, 1].set_xlabel("Feature Index [0-31]")
    axes[0, 1].legend()

    # (c) 59D uLBP Histogram
    axes[1, 0].bar(range(59), sample_lbp_hist, color="seagreen", alpha=0.75, width=0.8)
    axes[1, 0].set_title(f"(c) 59D Uniform LBP Texture Descriptor (P=8, R=1)")
    axes[1, 0].set_xlabel("LBP Pattern Bin Index [0-58]")

    # (d) Scree Plot
    axes[1, 1].plot(range(1, len(cum_var) + 1), cum_var, "o-", color="navy", linewidth=2)
    axes[1, 1].axhline(0.95, color="crimson", linestyle="--", label="95% Variance Threshold")
    axes[1, 1].axvline(n_pca, color="green", linestyle=":", label=f"Optimal ({n_pca}D)")
    axes[1, 1].set_title(f"(d) PCA Scree Plot: 95D -> {n_pca}D ({cum_var[-1]*100:.1f}% Variance)")
    axes[1, 1].set_xlabel("Principal Component Count")
    axes[1, 1].set_ylabel("Cumulative Explained Variance")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", out_filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved Stage 4 diagnostic: {out_path}")


def run_stage4():
    print("=" * 60)
    print("Machine Vision (UFMFRR-15-M) - Stage 4: 95D Features & PCA")
    print("=" * 60)

    # 1. Harvest & Process Apples
    apple_files = (
        glob.glob("minneapple-1-1/train/*.jpg")
        or glob.glob("/content/minneapple-1-1/train/*.jpg")
        or glob.glob("**/train/*.jpg", recursive=True)
    )
    if apple_files:
        apple_crops, apple_masks = harvest_apple_proposals(apple_files, max_images=5)
        if len(apple_crops) >= 10:
            evaluate_and_plot_task("Apple", apple_crops, apple_masks, "stage4_apple_output.png")

    # 2. Harvest & Process Real Resistors
    resistor_files = (
        glob.glob("Resistor Dataset/*.jpg")
        or glob.glob("Resistor Dataset/*.png")
        or glob.glob("/content/Resistor Dataset/*.jpg")
        or glob.glob("**/Resistor Dataset/*.jpg", recursive=True)
    )
    if resistor_files:
        res_crops, res_masks = harvest_resistor_proposals(resistor_files, max_images=5)
        if len(res_crops) >= 10:
            evaluate_and_plot_task("Resistor", res_crops, res_masks, "stage4_resistor_output.png")

    print("\nStage 4 completed successfully across both tasks!")
    print("=" * 60)


if __name__ == "__main__":
    run_stage4()
