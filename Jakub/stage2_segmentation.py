"""
Machine Vision (UFMFRR-15-M) - Group Project
Stage 2: Foreground Segmentation (Apples: Otsu vs. K-Means)

Primary Contributor: Jakub Wrzyszcz

Methodology:
  1. Feature Fusion:
     Combines CIE LAB a* (red-green chromaticity, high for ripe apples, low for leaves)
     and HSV Saturation S (vibrancy of fruit against bark/sky).
     A weighted scalar map M = alpha * a_norm + (1 - alpha) * s_norm is generated.
  2. Otsu Binarisation:
     Calculates the optimal global threshold on M that maximizes between-class variance.
  3. K-Means Clustering Benchmark:
     Partitions pixels in 3D (S, a*, b*) space into k=3 clusters (Fruit, Leaves, Background).
     The fruit cluster is identified as having the highest a* centroid.
  4. Quantitative Evaluation:
     Benchmarks both approaches on execution latency (ms) and spatial agreement (IoU / Dice).
"""

import os
import glob
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_stage1_lite(rgb_img):
    """Applies lightweight Stage 1 CLAHE on Luminance + spatial smoothing."""
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_chan)
    eq_rgb = cv2.cvtColor(cv2.merge([l_eq, a_chan, b_chan]), cv2.COLOR_LAB2RGB)

    denoised_rgb = cv2.medianBlur(eq_rgb, 3)
    clean_rgb = cv2.GaussianBlur(denoised_rgb, (3, 3), 0.8)
    return clean_rgb


def segment_apples_otsu(rgb_img, alpha=0.6):
    """
    Segments fruit by fusing CIE LAB a* and HSV Saturation S, followed by Otsu thresholding.

    Args:
        rgb_img: np.ndarray, RGB image.
        alpha: float, weighting factor for a* chromaticity (default 0.6).

    Returns:
        mask_otsu: np.ndarray, binary mask (0 or 255).
        fused_map: np.ndarray, normalized scalar combination map (0-255).
        threshold: float, optimal Otsu threshold value.
    """
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    a_chan = lab[:, :, 1].astype(np.float32)
    s_chan = hsv[:, :, 1].astype(np.float32)

    # In CIE LAB, a* > 128 indicates red/magenta; a* < 128 indicates green.
    # Penalize green leaf chromaticity by thresholding relative to neutral gray (120-128).
    a_diff = np.clip(a_chan - 120.0, 0.0, None)

    # Normalize a* and S to [0, 255]
    a_norm = cv2.normalize(a_diff, None, 0, 255, cv2.NORM_MINMAX)
    s_norm = cv2.normalize(s_chan, None, 0, 255, cv2.NORM_MINMAX)

    # Fused scalar feature map
    fused_map = (alpha * a_norm + (1.0 - alpha) * s_norm).astype(np.uint8)

    # Global Otsu Thresholding (maximizing between-class variance)
    threshold, mask_otsu = cv2.threshold(
        fused_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return mask_otsu, fused_map, threshold


def segment_apples_kmeans(rgb_img, k=3):
    """
    Segments fruit using k-Means clustering in 3D (S, a*, b*) feature space.

    Args:
        rgb_img: np.ndarray, RGB image.
        k: int, number of clusters (default 3: Fruit, Leaves, Background).

    Returns:
        mask_kmeans: np.ndarray, binary mask (0 or 255).
        centers: np.ndarray, cluster centroids in (S, a*, b*) space.
    """
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    s_chan = hsv[:, :, 1]
    a_chan = lab[:, :, 1]
    b_chan = lab[:, :, 2]

    h, w = rgb_img.shape[:2]
    # Build 3D feature matrix (N, 3)
    features = np.column_stack([s_chan.ravel(), a_chan.ravel(), b_chan.ravel()]).astype(
        np.float32
    )

    # Termination criteria: max 20 iterations or epsilon 0.2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
    _, labels, centers = cv2.kmeans(
        features, k, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS
    )

    # In orchard scenes, apple pixels exhibit the highest a* (redness) centroid
    apple_cluster_idx = int(np.argmax(centers[:, 1]))

    mask_kmeans = (labels.reshape((h, w)) == apple_cluster_idx).astype(np.uint8) * 255
    return mask_kmeans, centers


def benchmark_segmentation(rgb_img):
    """
    Executes and benchmarks Otsu thresholding vs K-Means clustering.

    Computes:
      - Execution time / latency (milliseconds).
      - Intersection over Union (IoU / Jaccard Index).
      - Dice Similarity Coefficient.
    """
    # 1. Benchmark Otsu
    t0 = time.perf_counter()
    mask_otsu, fused_map, thresh_val = segment_apples_otsu(rgb_img)
    t_otsu = (time.perf_counter() - t0) * 1000.0

    # 2. Benchmark K-Means
    t1 = time.perf_counter()
    mask_kmeans, centers = segment_apples_kmeans(rgb_img, k=3)
    t_kmeans = (time.perf_counter() - t1) * 1000.0

    # 3. Spatial Agreement Metrics
    bin_otsu = mask_otsu > 0
    bin_kmeans = mask_kmeans > 0

    intersection = np.logical_and(bin_otsu, bin_kmeans).sum()
    union = np.logical_or(bin_otsu, bin_kmeans).sum()

    iou = float(intersection) / float(union + 1e-8)
    dice = (2.0 * float(intersection)) / float(bin_otsu.sum() + bin_kmeans.sum() + 1e-8)

    print("\n" + "=" * 60)
    print("STAGE 2 BENCHMARK RESULTS: OTSU vs. K-MEANS")
    print("=" * 60)
    print(f"Otsu Latency:     {t_otsu:.2f} ms  (Threshold: {thresh_val:.1f})")
    print(f"K-Means Latency:  {t_kmeans:.2f} ms")
    print(f"Speedup:          {t_kmeans / max(t_otsu, 1e-5):.1f}x faster with Otsu")
    print(f"Intersection/Union (IoU): {iou:.4f}")
    print(f"Dice Coefficient:         {dice:.4f}")
    print(f"Otsu FG Pixel Count:      {bin_otsu.sum():,} ({bin_otsu.mean() * 100:.2f}%)")
    print(f"K-Means FG Pixel Count:   {bin_kmeans.sum():,} ({bin_kmeans.mean() * 100:.2f}%)")
    print("=" * 60 + "\n")

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


def run_stage2():
    print("=" * 60)
    print("Machine Vision (UFMFRR-15-M) - Stage 2: Foreground Segmentation")
    print("=" * 60)

    img_files = (
        glob.glob("minneapple-1-1/train/*.jpg")
        or glob.glob("/content/minneapple-1-1/train/*.jpg")
        or glob.glob("**/train/*.jpg", recursive=True)
    )

    if not img_files:
        print("[ERROR] Could not find dataset images in minneapple-1-1/train/.")
        return

    img_path = img_files[0]
    raw_bgr = cv2.imread(img_path)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    print(f"Loaded: {os.path.basename(img_path)} (Shape: {raw_rgb.shape})")

    # Preprocess via Stage 1 pipeline
    clean_rgb = preprocess_stage1_lite(raw_rgb)

    # Benchmark Otsu vs K-Means
    bench = benchmark_segmentation(clean_rgb)

    # 4-Panel Report Diagnostic Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Input RGB
    axes[0, 0].imshow(clean_rgb)
    axes[0, 0].set_title(f"(a) Input Image: {os.path.basename(img_path)}")
    axes[0, 0].axis("off")

    # (b) Fused Scalar Feature Map (a* + S)
    axes[0, 1].imshow(bench["fused_map"], cmap="magma")
    axes[0, 1].set_title(
        f"(b) Fused Feature Map [0.6 a* + 0.4 S] (Otsu Thresh = {bench['thresh_val']:.1f})"
    )
    axes[0, 1].axis("off")

    # (c) Otsu Binary Segmentation Mask
    axes[1, 0].imshow(bench["mask_otsu"], cmap="gray")
    axes[1, 0].set_title(
        f"(c) Otsu Segmentation Mask (Latency: {bench['t_otsu_ms']:.1f} ms)"
    )
    axes[1, 0].axis("off")

    # (d) K-Means Binary Segmentation Mask
    axes[1, 1].imshow(bench["mask_kmeans"], cmap="gray")
    axes[1, 1].set_title(
        f"(d) K-Means (k=3) Mask (Latency: {bench['t_kmeans_ms']:.1f} ms | IoU: {bench['iou']:.3f})"
    )
    axes[1, 1].axis("off")

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "stage2_output.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")

    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            plt.show()
        else:
            plt.close()
    except Exception:
        plt.close()

    print(f"Stage 2 successfully completed! Output saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_stage2()
