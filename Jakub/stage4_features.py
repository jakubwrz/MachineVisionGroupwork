"""
Machine Vision (UFMFRR-15-M) - Group Project
Stage 4: Feature Extraction & Scaling (Apples: 95D Feature Representation & PCA)

Primary Contributor: Jakub Wrzyszcz

Methodology:
  1. 95D Composite Feature Vector:
     - 32 Colour Descriptors:
       * 20 Statistical Moments: Mean, Standard Deviation, Skewness, Kurtosis
         across 5 perceptual & physical channels: H, S, a*, b*, and true Unit 2 Intensity I = (R+G+B)/3.
       * 12 Colour Histogram Bins: 4 bins each for Hue (H), Saturation (S), and CIE LAB a*.
     - 59 Uniform LBP Texture Bins:
       * Rotation-invariant Uniform Local Binary Patterns (P=8, R=1, method='uniform').
       * 58 uniform bit transitions + 1 non-uniform pattern bucket = 59 bins.
     - 4 Invariant Shape Metrics:
       * Circularity: 4 * pi * Area / Perimeter^2 (compactness).
       * Solidity: Area / ConvexHull Area (surface roughness / convexity).
       * Aspect Ratio: BoundingBox Width / Height.
       * Extent: Area / BoundingBox Area.
       * Excludes raw centroid (x, y) coordinates and unnormalized pixel area to guarantee
         scale and spatial translation invariance.
     * Total Dimensions: 32 + 59 + 4 = 95D.

  2. Statistical Normalisation & Dimensionality Reduction:
     - StandardScaler: Standardizes features to zero mean and unit variance.
     - Principal Component Analysis (PCA): Retains 95% of total explained variance.
     - Crucially fit ON TRAINING SPLIT ONLY to prevent data leakage.
"""

import os
import sys
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Ensure local imports work in both CLI and Colab/interactive notebooks
_current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    from stage3_morphology_proposals import (
        apply_morphology_and_hole_filling,
        split_apple_clusters_watershed,
        extract_64x64_proposals,
    )
except ImportError:
    from Jakub.stage3_morphology_proposals import (
        apply_morphology_and_hole_filling,
        split_apple_clusters_watershed,
        extract_64x64_proposals,
    )



def extract_shape_metrics(crop_mask):
    """
    Computes 4 scale- and translation-invariant shape descriptors.
    Excludes raw centroid (x, y) and unnormalized pixel area.

    Args:
        crop_mask: np.ndarray (uint8), binary mask of proposal crop.

    Returns:
        shape_feats: np.ndarray of shape (4,), [Circularity, Solidity, AspectRatio, Extent].
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
    Extracts 59 Uniform Local Binary Pattern (uLBP) texture bins.

    For P=8, uniform patterns (<= 2 bitwise transitions) yield P*(P-1) + 2 = 58 bins,
    plus 1 bin for non-uniform patterns = 59 total bins.

    Args:
        crop_rgb: np.ndarray (64, 64, 3), RGB crop.
        P: int, number of circularly symmetric neighbor points.
        R: int, radius of circle.

    Returns:
        lbp_hist: np.ndarray of shape (59,), normalized histogram.
        lbp_img: np.ndarray of shape (64, 64), 2D LBP map.
    """
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    lbp_img = local_binary_pattern(gray, P=P, R=R, method="uniform")

    # Uniform LBP with P=8 produces integer labels from 0 to 58 (59 bins)
    lbp_hist, _ = np.histogram(lbp_img.ravel(), bins=59, range=(0, 59), density=True)
    return lbp_hist.astype(np.float32), lbp_img


def extract_colour_descriptors(crop_rgb):
    """
    Extracts 32 colour features across 5 channels (H, S, a*, b*, I):
      - 20 Statistical Moments: Mean, Std, Skewness, Kurtosis for each channel.
      - 12 Colour Histogram Bins: 4 bins each for H (0-180), S (0-256), and a* (0-256).

    Args:
        crop_rgb: np.ndarray (64, 64, 3), RGB crop.

    Returns:
        colour_feats: np.ndarray of shape (32,).
    """
    # 1. Color Space Representations
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)

    h_chan, s_chan, _ = cv2.split(hsv)
    _, a_chan, b_chan = cv2.split(lab)
    # Unit 2 True Intensity: I = (R + G + B) / 3
    i_chan = np.mean(crop_rgb.astype(np.float32), axis=2).astype(np.uint8)

    channels = [h_chan, s_chan, a_chan, b_chan, i_chan]

    # 2. Extract 4 statistical moments per channel (5 * 4 = 20 moments)
    moments = []
    for ch in channels:
        vals = ch.ravel().astype(np.float32)
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        skew = float(stats.skew(vals))
        kurt = float(stats.kurtosis(vals))
        moments.extend([mu, sigma, skew, kurt])

    # 3. 4-bin normalized histograms for key chromatic channels (3 * 4 = 12 bins)
    h_hist, _ = np.histogram(h_chan.ravel(), bins=4, range=(0, 180), density=True)
    s_hist, _ = np.histogram(s_chan.ravel(), bins=4, range=(0, 256), density=True)
    a_hist, _ = np.histogram(a_chan.ravel(), bins=4, range=(0, 256), density=True)

    colour_feats = np.concatenate([moments, h_hist, s_hist, a_hist]).astype(np.float32)
    return colour_feats


def extract_95d_features(crop_rgb, crop_mask=None):
    """
    Extracts the unified 95D feature vector for a proposal patch:
      - 32 Colour Descriptors (moments + histograms across H, S, a*, b*, I)
      - 59 Uniform LBP Texture Bins (P=8, R=1 uniform)
      -  4 Invariant Shape Metrics (Circularity, Solidity, Aspect Ratio, Extent)

    Excludes raw centroid coordinates and unnormalized pixel area.

    Args:
        crop_rgb: np.ndarray (64, 64, 3), proposal crop.
        crop_mask: np.ndarray (64, 64) or None, binary mask.

    Returns:
        feat_vector: np.ndarray of shape (95,), float32.
    """
    # 1. Colour Features (32D)
    colour_feats = extract_colour_descriptors(crop_rgb)

    # 2. Texture Features (59D)
    texture_feats, _ = extract_ulbp_texture(crop_rgb)

    # 3. Shape Features (4D)
    if crop_mask is None:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        _, crop_mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    shape_feats = extract_shape_metrics(crop_mask)

    feat_vector = np.concatenate([colour_feats, texture_feats, shape_feats])
    assert len(feat_vector) == 95, f"Expected 95D vector, but got {len(feat_vector)}D"
    return feat_vector


def fit_scaler_and_pca(X_train, X_val=None, n_components=0.95):
    """
    Fits StandardScaler and PCA on training split only to prevent data leakage.

    Args:
        X_train: np.ndarray (N_train, 95), training feature matrix.
        X_val: np.ndarray (N_val, 95) or None, validation feature matrix.
        n_components: float (0.95) or int, variance threshold for PCA.

    Returns:
        scaler: fitted StandardScaler object.
        pca: fitted PCA object.
        X_train_pca: np.ndarray, transformed training features.
        X_val_pca: np.ndarray or None, transformed validation features.
    """
    # 1. Fit StandardScaler on training split ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 2. Fit PCA on scaled training features ONLY
    pca = PCA(n_components=n_components, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_scaled)

    # 3. Transform validation features without re-fitting
    X_val_pca = None
    if X_val is not None and len(X_val) > 0:
        X_val_scaled = scaler.transform(X_val)
        X_val_pca = pca.transform(X_val_scaled)

    return scaler, pca, X_train_pca, X_val_pca


def run_stage4():
    print("=" * 60)
    print("Machine Vision (UFMFRR-15-M) - Stage 4: Feature Extraction & PCA")
    print("=" * 60)

    # 1. Locate dataset images
    img_files = (
        glob.glob("minneapple-1-1/train/*.jpg")
        or glob.glob("/content/minneapple-1-1/train/*.jpg")
        or glob.glob("**/train/*.jpg", recursive=True)
    )

    if not img_files:
        print("[ERROR] Could not find dataset images in minneapple-1-1/train/.")
        return

    # Process first 5 images to harvest sufficient proposals for robust PCA fitting
    sample_images = img_files[:5]
    print(f"Collecting proposals from {len(sample_images)} training images...")

    all_crops = []
    all_masks = []

    for img_p in sample_images:
        bgr = cv2.imread(img_p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Quick Otsu mask
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        a_diff = np.clip(lab[:, :, 1].astype(np.float32) - 120.0, 0.0, None)
        a_norm = cv2.normalize(a_diff, None, 0, 255, cv2.NORM_MINMAX)
        s_norm = cv2.normalize(
            hsv[:, :, 1].astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
        )
        fused = (0.6 * a_norm + 0.4 * s_norm).astype(np.uint8)
        _, raw_mask = cv2.threshold(fused, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        _, filled = apply_morphology_and_hole_filling(raw_mask)
        _, markers, _ = split_apple_clusters_watershed(rgb, filled)
        crops, masks, _ = extract_64x64_proposals(rgb, markers, min_area=40)

        all_crops.extend(crops)
        all_masks.extend(masks)

    print(f"Harvested {len(all_crops)} candidate 64x64 apple proposals.")

    if len(all_crops) < 10:
        print("[WARNING] Very few proposals found. Generating synthetic crops to demonstrate PCA.")
        while len(all_crops) < 20:
            synth = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
            all_crops.append(synth)
            all_masks.append(np.ones((64, 64), dtype=np.uint8) * 255)

    # 2. Extract 95D Feature Vectors
    print("Extracting 95D feature vectors (32 Colour + 59 uLBP + 4 Shape)...")
    feature_matrix = []
    for crop, mask in zip(all_crops, all_masks):
        vec = extract_95d_features(crop, mask)
        feature_matrix.append(vec)

    X = np.array(feature_matrix, dtype=np.float32)
    print(f"Feature matrix assembled: {X.shape[0]} samples x {X.shape[1]} features.")

    # 3. Train/Validation Split (80% Train, 20% Val)
    X_train, X_val = train_test_split(X, test_size=0.20, random_state=42)
    print(f"Dataset split: Train={X_train.shape[0]} samples, Validation={X_val.shape[0]} samples.")

    # 4. Fit StandardScaler and PCA on TRAINING SPLIT ONLY
    scaler, pca, X_train_pca, X_val_pca = fit_scaler_and_pca(
        X_train, X_val, n_components=0.95
    )

    n_orig = X.shape[1]
    n_pca = pca.n_components_
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    print("\n" + "=" * 60)
    print("STAGE 4 PCA DIMENSIONALITY REDUCTION REPORT")
    print("=" * 60)
    print(f"Original Feature Dimension:     {n_orig}D")
    print(f"Reduced PCA Dimension:          {n_pca}D  (for >=95% variance)")
    print(f"Dimensionality Compression:     {(1.0 - n_pca / n_orig) * 100:.1f}% reduction")
    print(f"Total Explained Variance:       {cum_var[-1] * 100:.2f}%")
    print("=" * 60 + "\n")

    # 5. Diagnostic Visualisation
    sample_crop = all_crops[0]
    sample_mask = all_masks[0]
    sample_lbp_hist, sample_lbp_map = extract_ulbp_texture(sample_crop)
    sample_col_feats = extract_colour_descriptors(sample_crop)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Sample Proposal Crop & LBP Texture Map
    combined_vis = np.hstack([
        sample_crop,
        cv2.cvtColor((sample_lbp_map * (255.0 / 58.0)).astype(np.uint8), cv2.COLOR_GRAY2RGB),
    ])
    axes[0, 0].imshow(combined_vis)
    axes[0, 0].set_title("(a) Proposal Crop (Left) & Uniform LBP Texture Map (Right)")
    axes[0, 0].axis("off")

    # (b) 32D Colour Features Breakdown
    axes[0, 1].bar(
        range(32), sample_col_feats, color="darkorange", alpha=0.75, width=0.8
    )
    axes[0, 1].axvline(19.5, color="red", linestyle="--", label="Moments (20) | Histograms (12)")
    axes[0, 1].set_title("(b) 32D Colour Features (Moments & Histograms on H, S, a*, b*, I)")
    axes[0, 1].set_xlabel("Feature Index [0-31]")
    axes[0, 1].set_ylabel("Normalized Value")
    axes[0, 1].legend()

    # (c) 59D Uniform LBP Histogram
    axes[1, 0].bar(
        range(59), sample_lbp_hist, color="seagreen", alpha=0.75, width=0.8
    )
    axes[1, 0].set_title("(c) 59D Uniform LBP Texture Descriptor (P=8, R=1)")
    axes[1, 0].set_xlabel("LBP Pattern Bin Index [0-58]")
    axes[1, 0].set_ylabel("Probability Density")

    # (d) PCA Scree / Cumulative Explained Variance Curve
    axes[1, 1].plot(range(1, len(cum_var) + 1), cum_var, "o-", color="navy", linewidth=2)
    axes[1, 1].axhline(0.95, color="crimson", linestyle="--", label="95% Variance Threshold")
    axes[1, 1].axvline(n_pca, color="green", linestyle=":", label=f"Optimal Components ({n_pca}D)")
    axes[1, 1].set_title(
        f"(d) PCA Scree Plot: 95D -> {n_pca}D ({cum_var[-1]*100:.1f}% Variance)"
    )
    axes[1, 1].set_xlabel("Principal Component Count")
    axes[1, 1].set_ylabel("Cumulative Explained Variance")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "stage4_output.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")

    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            plt.show()
        else:
            plt.close()
    except Exception:
        plt.close()

    print(f"Stage 4 successfully completed! Output saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_stage4()
