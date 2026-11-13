"""
Machine Vision (UFMFRR-15-M) - Group Project
Stage 2: Foreground Segmentation (Apples & Resistors)

Primary Contributor: Jakub Wrzyszcz

Unified Pipeline Methodology:
  Task 1 (Orchard Apples):
    1. Feature Fusion: Combines CIE LAB a* (red-green chromaticity, high for ripe apples,
       low for foliage) and HSV Saturation S (chromatic vibrancy against sky/bark).
       M = 0.6 * a_norm + 0.4 * s_norm.
    2. Otsu Binarisation: Determines optimal global threshold maximizing between-class variance.
    3. K-Means Benchmark: Evaluates against cv2.kmeans (k=3) in 3D (S, a*, b*) space, demonstrating
       computational speedup and spatial precision.

  Task 2 (Multi-Resistor Scenario):
    1. Adaptive Thresholding (Unit 4): Applies cv2.adaptiveThreshold on inverted intensity/grayscale
       (ADAPTIVE_THRESH_GAUSSIAN_C) to isolate dark resistor bodies, wire leads, and bands
       under non-uniform workbench illumination.
    2. Morphological Closing (cv2.MORPH_CLOSE, 5x5 rectangular structuring element):
       Bridges internal gaps across dark resistor colour bands and specular glare streaks,
       preventing a single resistor from fragmenting into multiple disjoint blobs.
    3. Connected Component Area Filtering: Filters candidate blobs to retain solid resistor
       bodies with area > 500 pixels.
"""

import os
import glob
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ensure local imports work across environments
_curr_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
import sys
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

try:
    from stage1_preprocessing import preprocess_stage1
except ImportError:
    from Jakub.stage1_preprocessing import preprocess_stage1


# ==============================================================================
# TASK 1: APPLE SEGMENTATION (OTSU vs. K-MEANS)
# ==============================================================================

def segment_apples_otsu(rgb_img, alpha=0.6):
    """
    Segments fruit by fusing CIE LAB a* and HSV Saturation S, followed by Otsu thresholding.
    """
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    a_chan = lab[:, :, 1].astype(np.float32)
    s_chan = hsv[:, :, 1].astype(np.float32)

    # In CIE LAB, a* > 128 indicates red/magenta; penalize green foliage (a* < 120)
    a_diff = np.clip(a_chan - 120.0, 0.0, None)

    a_norm = cv2.normalize(a_diff, None, 0, 255, cv2.NORM_MINMAX)
    s_norm = cv2.normalize(s_chan, None, 0, 255, cv2.NORM_MINMAX)

    # Fused scalar feature map
    fused_map = (alpha * a_norm + (1.0 - alpha) * s_norm).astype(np.uint8)

    # Global Otsu Thresholding
    threshold, mask_otsu = cv2.threshold(
        fused_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return mask_otsu, fused_map, threshold


def segment_apples_kmeans(rgb_img, k=3):
    """
    Segments fruit using k-Means clustering in 3D (S, a*, b*) feature space.
    """
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    s_chan = hsv[:, :, 1]
    a_chan = lab[:, :, 1]
    b_chan = lab[:, :, 2]

    h, w = rgb_img.shape[:2]
    features = np.column_stack([s_chan.ravel(), a_chan.ravel(), b_chan.ravel()]).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
    _, labels, centers = cv2.kmeans(
        features, k, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS
    )

    # Apple cluster exhibits highest a* centroid
    apple_cluster_idx = int(np.argmax(centers[:, 1]))
    mask_kmeans = (labels.reshape((h, w)) == apple_cluster_idx).astype(np.uint8) * 255
    return mask_kmeans, centers


def benchmark_apple_segmentation(rgb_img):
    """Benchmarks Otsu vs K-Means on execution time and spatial agreement (IoU/Dice)."""
    t0 = time.perf_counter()
    mask_otsu, fused_map, thresh_val = segment_apples_otsu(rgb_img)
    t_otsu = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    mask_kmeans, _ = segment_apples_kmeans(rgb_img, k=3)
    t_kmeans = (time.perf_counter() - t1) * 1000.0

    bin_otsu = mask_otsu > 0
    bin_kmeans = mask_kmeans > 0
    intersection = np.logical_and(bin_otsu, bin_kmeans).sum()
    union = np.logical_or(bin_otsu, bin_kmeans).sum()

    iou = float(intersection) / float(union + 1e-8)
    dice = (2.0 * float(intersection)) / float(bin_otsu.sum() + bin_kmeans.sum() + 1e-8)

    print("\n" + "-" * 55)
    print("STAGE 2 APPLE BENCHMARK: OTSU vs. K-MEANS")
    print("-" * 55)
    print(f"Otsu Latency:     {t_otsu:.2f} ms  (Threshold: {thresh_val:.1f})")
    print(f"K-Means Latency:  {t_kmeans:.2f} ms")
    print(f"Speedup:          {t_kmeans / max(t_otsu, 1e-5):.1f}x faster with Otsu")
    print(f"IoU Overlap:      {iou:.4f}")
    print(f"Dice Coefficient: {dice:.4f}")
    print("-" * 55)

    return {
        "mask_otsu": mask_otsu,
        "mask_kmeans": mask_kmeans,
        "fused_map": fused_map,
        "thresh_val": thresh_val,
        "t_otsu_ms": t_otsu,
        "t_kmeans_ms": t_kmeans,
        "iou": iou,
        "dice": dice,
    }


# ==============================================================================
# TASK 2: RESISTOR BODY SEGMENTATION (ADAPTIVE THRESHOLDING & MORPH CLOSE)
# ==============================================================================

def segment_resistors_adaptive(rgb_img, min_area=500):
    """
    Segments resistors via adaptive thresholding and 5x5 morphological closing.

    Args:
        rgb_img: np.ndarray, RGB image.
        min_area: int, minimum blob area threshold (default 500 pixels).

    Returns:
        body_mask: np.ndarray (uint8), filtered binary mask of resistor bodies.
        raw_thresh: np.ndarray, pre-closing thresholded mask.
        blobs: list of contour arrays corresponding to filtered resistor bodies.
    """
    # 1. Convert to grayscale and apply mild Gaussian blur
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Adaptive Gaussian Thresholding (inverted: dark components -> white foreground)
    raw_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )

    # 3. 5x5 Morphological Closing (cv2.MORPH_CLOSE) to bridge dark color bands
    kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(raw_thresh, cv2.MORPH_CLOSE, kernel_5x5, iterations=2)

    # 4. Filter blobs with area > 500 pixels
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(closed)
    resistor_blobs = []

    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            cv2.drawContours(filtered_mask, [c], -1, 255, thickness=cv2.FILLED)
            resistor_blobs.append(c)

    return filtered_mask, raw_thresh, resistor_blobs


# ==============================================================================
# UNIFIED INTERFACE & EXECUTION
# ==============================================================================

def segment_foreground(rgb_img, mode="apple"):
    """
    Unified Stage 2 entry point.
      - mode='apple': returns binary mask from fused a*+S Otsu thresholding.
      - mode='resistor': returns binary mask from adaptive thresholding + 5x5 closing.
    """
    if mode == "apple":
        mask, _, _ = segment_apples_otsu(rgb_img)
        return mask
    elif mode == "resistor":
        mask, _, _ = segment_resistors_adaptive(rgb_img)
        return mask
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'apple' or 'resistor'.")


def run_stage2():
    print("=" * 60)
    print("Machine Vision (UFMFRR-15-M) - Stage 2: Foreground Segmentation")
    print("=" * 60)

    os.makedirs("output", exist_ok=True)

    # ---------------------------------------------------------
    # 1. Process Apple Scenario
    # ---------------------------------------------------------
    apple_files = (
        glob.glob("minneapple-1-1/train/*.jpg")
        or glob.glob("/content/minneapple-1-1/train/*.jpg")
        or glob.glob("**/train/*.jpg", recursive=True)
    )
    if apple_files:
        apple_bgr = cv2.imread(apple_files[0])
        apple_rgb = cv2.cvtColor(apple_bgr, cv2.COLOR_BGR2RGB)
        clean_apple, _, _, _, _, _, _ = preprocess_stage1(apple_rgb, mode="apple")
        bench = benchmark_apple_segmentation(clean_apple)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].imshow(clean_apple)
        axes[0, 0].set_title(f"(a) Preprocessed Apple RGB: {os.path.basename(apple_files[0])}")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(bench["fused_map"], cmap="magma")
        axes[0, 1].set_title(f"(b) Fused a*+S Feature Map (Otsu Thresh = {bench['thresh_val']:.1f})")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(bench["mask_otsu"], cmap="gray")
        axes[1, 0].set_title(f"(c) Otsu Binary Mask (Latency: {bench['t_otsu_ms']:.1f} ms)")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(bench["mask_kmeans"], cmap="gray")
        axes[1, 1].set_title(f"(d) K-Means (k=3) Mask (Latency: {bench['t_kmeans_ms']:.1f} ms | IoU: {bench['iou']:.3f})")
        axes[1, 1].axis("off")

        plt.tight_layout()
        out_apple = os.path.join("output", "stage2_apple_output.png")
        plt.savefig(out_apple, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[APPLE] Saved Stage 2 diagnostic: {out_apple}")

    # ---------------------------------------------------------
    # 2. Process Resistor Scenario
    # ---------------------------------------------------------
    resistor_files = (
        glob.glob("Resistor Dataset/*.jpg")
        or glob.glob("Resistor Dataset/*.png")
        or glob.glob("/content/Resistor Dataset/*.jpg")
        or glob.glob("**/Resistor Dataset/*.jpg", recursive=True)
    )
    if resistor_files:
        res_bgr = cv2.imread(resistor_files[0])
        h, w = res_bgr.shape[:2]
        if max(h, w) > 1280:
            scale = 1280.0 / max(h, w)
            res_bgr = cv2.resize(res_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
        clean_res, _, _, _, _, _, _ = preprocess_stage1(res_rgb, mode="resistor")

        t0 = time.perf_counter()
        body_mask, raw_thresh, resistor_blobs = segment_resistors_adaptive(clean_res, min_area=500)
        t_res = (time.perf_counter() - t0) * 1000.0

        print(f"\n[RESISTOR] Segmented {len(resistor_blobs)} resistor bodies in {t_res:.2f} ms")

        # Visual overlay with contour bounding boxes
        overlay = clean_res.copy()
        for c in resistor_blobs:
            x, y, bw, bh = cv2.boundingRect(c)
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.drawContours(overlay, [c], -1, (255, 0, 0), 2)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].imshow(clean_res)
        axes[0, 0].set_title(f"(a) Input Resistor RGB: {os.path.basename(resistor_files[0])}")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(raw_thresh, cmap="gray")
        axes[0, 1].set_title("(b) Adaptive Thresholding (Pre-closing)")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(body_mask, cmap="gray")
        axes[1, 0].set_title(f"(c) 5x5 Morphological Closing (Area > 500 px, Latency: {t_res:.1f} ms)")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title(f"(d) Segmented Resistor Bodies ({len(resistor_blobs)} detected)")
        axes[1, 1].axis("off")

        plt.tight_layout()
        out_res = os.path.join("output", "stage2_resistor_output.png")
        plt.savefig(out_res, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[RESISTOR] Saved Stage 2 diagnostic: {out_res}")

    print("\nStage 2 completed successfully across both tasks!")
    print("=" * 60)


if __name__ == "__main__":
    run_stage2()
