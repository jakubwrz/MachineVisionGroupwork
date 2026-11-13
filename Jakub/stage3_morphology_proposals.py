"""
Machine Vision (UFMFRR-15-M) - Group Project
Stage 3: Morphology, Pose & Proposals (Apples: Watershed Splitting)

Primary Contributor: Jakub Wrzyszcz

Methodology:
  1. Morphological Refinement:
     - 3x3 Opening (cv2.MORPH_OPEN) eliminates isolated false-positive noise and thin spurs.
     - Hole filling via contour hierarchy fills internal dark spots / specular reflections.
  2. Euclidean Distance Transform (cv2.distanceTransform, cv2.DIST_L2):
     - Computes the metric distance from each foreground pixel to the nearest background boundary.
     - Touching and overlapping fruit clusters display distinct topological peaks corresponding to individual centres.
  3. Marker-Controlled Watershed (cv2.watershed):
     - Peaks thresholded at a fraction of max distance serve as sure foreground seeds.
     - Dilated masks define sure background; unassigned boundaries are flooded along gradient lines.
  4. Instance Bounding & Proposal Cropping:
     - Extracts candidate fruit instances, calculates centroids and bounding boxes,
       and exports standard 64x64 RGB crops and binary masks ready for Stage 4 feature extraction.
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_morphology_and_hole_filling(binary_mask):
    """
    Applies 3x3 morphological opening and topological hole-filling.

    Args:
        binary_mask: np.ndarray (uint8), binary mask (0 or 255).

    Returns:
        opened: np.ndarray, post-opening mask.
        filled: np.ndarray, hole-filled binary mask.
    """
    # 1. 3x3 Morphological Opening (Unit 5 Slides 30-36)
    kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_3x3)

    # 2. Topological Hole Filling
    # External contours are traced and drawn filled, sealing internal specular cavities
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(opened)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    return opened, filled


def split_apple_clusters_watershed(rgb_img, filled_mask, peak_thresh_ratio=0.30):
    """
    Performs Euclidean Distance Transform and Marker-Controlled Watershed
    to split clustered and overlapping apples into distinct instances.

    Args:
        rgb_img: np.ndarray, RGB image.
        filled_mask: np.ndarray (uint8), clean binary mask.
        peak_thresh_ratio: float, distance peak threshold factor relative to maximum.

    Returns:
        dist_transform: np.ndarray, Euclidean Distance Transform map.
        markers: np.ndarray (int32), watershed label matrix.
        num_instances: int, number of separated fruit instances.
    """
    # 1. Euclidean Distance Transform (L2 metric, 5x5 mask)
    dist_transform = cv2.distanceTransform(filled_mask, cv2.DIST_L2, 5)

    # 2. Sure Foreground Seeds (topological distance crests)
    max_dist = dist_transform.max()
    if max_dist > 0:
        _, sure_fg = cv2.threshold(
            dist_transform, peak_thresh_ratio * max_dist, 255, cv2.THRESH_BINARY
        )
    else:
        sure_fg = np.zeros_like(filled_mask)
    sure_fg = sure_fg.astype(np.uint8)

    # 3. Sure Background (dilated boundary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sure_bg = cv2.dilate(filled_mask, kernel, iterations=3)

    # 4. Unknown boundary region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 5. Connected components labelling on seeds
    num_markers, markers = cv2.connectedComponents(sure_fg)

    # Marker shift: background = 1, unknown boundary = 0, instances >= 2
    markers = markers + 1
    markers[unknown == 255] = 0

    # 6. Marker-Controlled Watershed flooding
    # Watershed expects a 3-channel BGR or RGB image
    watershed_img = rgb_img.copy()
    markers = cv2.watershed(watershed_img, markers)

    num_instances = max(0, num_markers - 1)
    return dist_transform, markers, num_instances


def extract_64x64_proposals(rgb_img, markers, min_area=50, target_size=(64, 64)):
    """
    Extracts individual apple proposal patches resized to target_size (64x64x3).

    Args:
        rgb_img: np.ndarray, RGB image.
        markers: np.ndarray (int32), watershed label matrix.
        min_area: int, minimum pixel area threshold to discard noise.
        target_size: tuple, (width, height) standard proposal dimension.

    Returns:
        crops: list of np.ndarray (64, 64, 3), RGB crops.
        crop_masks: list of np.ndarray (64, 64), binary masks corresponding to the crop.
        metadata: list of dicts with centroid (cx, cy), bounding box (x, y, w, h), and area.
    """
    h_img, w_img = rgb_img.shape[:2]
    unique_labels = [label for label in np.unique(markers) if label >= 2]

    crops = []
    crop_masks = []
    metadata = []

    for label in unique_labels:
        inst_mask = (markers == label).astype(np.uint8) * 255
        area = int(np.sum(inst_mask > 0))

        if area < min_area:
            continue

        # Compute spatial moments for centroid
        M = cv2.moments(inst_mask)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Bounding box
        x, y, w, h = cv2.boundingRect(inst_mask)

        # Centered bounding square with padding
        side = max(w, h, 32)
        pad = int(side * 0.2)
        side_padded = side + 2 * pad
        half_side = side_padded // 2

        x1 = max(0, cx - half_side)
        x2 = min(w_img, cx + half_side)
        y1 = max(0, cy - half_side)
        y2 = min(h_img, cy + half_side)

        patch_rgb = rgb_img[y1:y2, x1:x2]
        patch_mask = inst_mask[y1:y2, x1:x2]

        if patch_rgb.shape[0] < 4 or patch_rgb.shape[1] < 4:
            continue

        # Standardise to 64x64 using area interpolation
        crop_64 = cv2.resize(patch_rgb, target_size, interpolation=cv2.INTER_AREA)
        mask_64 = cv2.resize(patch_mask, target_size, interpolation=cv2.INTER_NEAREST)

        crops.append(crop_64)
        crop_masks.append(mask_64)
        metadata.append({
            "label": label,
            "centroid": (cx, cy),
            "bbox": (x, y, w, h),
            "crop_coords": (x1, y1, x2, y2),
            "area": area,
        })

    return crops, crop_masks, metadata


def run_stage3():
    print("=" * 60)
    print("Machine Vision (UFMFRR-15-M) - Stage 3: Morphology & Proposals")
    print("=" * 60)

    # 1. Locate dataset image
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

    # 2. Stage 2 Quick Segmentation (a* + S Otsu)
    lab = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2HSV)
    a_diff = np.clip(lab[:, :, 1].astype(np.float32) - 120.0, 0.0, None)
    a_norm = cv2.normalize(a_diff, None, 0, 255, cv2.NORM_MINMAX)
    s_norm = cv2.normalize(hsv[:, :, 1].astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    fused = (0.6 * a_norm + 0.4 * s_norm).astype(np.uint8)
    _, raw_mask = cv2.threshold(fused, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Stage 3 Morphology & Hole Filling
    opened, filled = apply_morphology_and_hole_filling(raw_mask)

    # 4. Watershed Splitting via Euclidean Distance Transform
    dist_map, markers, n_inst = split_apple_clusters_watershed(raw_rgb, filled)
    print(f"Watershed splitting identified {n_inst} candidate instances.")

    # 5. Extract standard 64x64 proposals
    crops, crop_masks, meta = extract_64x64_proposals(raw_rgb, markers, min_area=50)
    print(f"Successfully extracted {len(crops)} standardized 64x64 crops.")

    # 6. Diagnostic Visualisation
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)

    # (a) Input RGB with overlay contours
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(raw_rgb)
    ax0.set_title(f"(a) Input: {os.path.basename(img_path)}")
    ax0.axis("off")

    # (b) Euclidean Distance Transform Heatmap
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(dist_map, cmap="jet")
    ax1.set_title("(b) Euclidean Distance Transform (L2)")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # (c) Watershed Instance Segmentation Overlay
    ax2 = fig.add_subplot(gs[0, 2])
    # Generate colored watershed overlay
    ws_vis = raw_rgb.copy()
    np.random.seed(42)
    colors = np.random.randint(0, 255, (max(markers.max() + 2, 10), 3), dtype=np.uint8)
    for m_id in np.unique(markers):
        if m_id >= 2:
            ws_vis[markers == m_id] = (
                0.5 * ws_vis[markers == m_id] + 0.5 * colors[m_id]
            ).astype(np.uint8)
    # Watershed boundaries (-1) highlighted in yellow
    ws_vis[markers == -1] = [255, 255, 0]
    ax2.imshow(ws_vis)
    ax2.set_title(f"(c) Watershed Instances ({len(crops)} crops exported)")
    ax2.axis("off")

    # (d) 64x64 Proposal Crops Grid (Bottom Row)
    ax3 = fig.add_subplot(gs[1, :])
    n_display = min(len(crops), 10)
    if n_display > 0:
        crop_strip = np.hstack([crops[i] for i in range(n_display)])
        ax3.imshow(crop_strip)
        ax3.set_title(
            f"(d) Exported 64x64 Normalized Proposal Crops (Showing first {n_display} of {len(crops)})"
        )
    else:
        ax3.text(0.5, 0.5, "No proposals met the area threshold", ha="center")
    ax3.axis("off")

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "stage3_output.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")

    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            plt.show()
        else:
            plt.close()
    except Exception:
        plt.close()

    print(f"Stage 3 successfully completed! Output saved to: {out_path}")
    print("=" * 60)
    return crops, crop_masks, meta


if __name__ == "__main__":
    run_stage3()
