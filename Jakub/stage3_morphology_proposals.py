"""
Machine Vision (UFMFRR-15-M) - Group Project
Stage 3: Morphology, Pose & Proposals (Apples & Resistors)

Primary Contributor: Jakub Wrzyszcz

Unified Pipeline Methodology:
  Task 1 (Orchard Apples - Watershed Splitting):
    1. 3x3 Opening (cv2.MORPH_OPEN) suppresses thin twigs and false-positive spurs.
    2. Hole filling via contour hierarchy seals internal specular reflection cavities.
    3. Euclidean Distance Transform (EDT, L2 metric) computes topological distance to background;
       local maxima identify individual fruit centers even within touching clusters.
    4. Marker-controlled cv2.watershed splits overlapping apples into distinct instances.
    5. Standardized 64x64 proposal crops and masks are exported.

  Task 2 (Multi-Resistor Scenario - Pose Alignment, Lead Stripping & Band Slicing):
    1. Horizontal Pose Alignment (cv2.minAreaRect & cv2.warpAffine):
       Computes the minimum-area bounding box for each resistor contour. Rotates the
       image and mask so the cylindrical body axis is strictly horizontal.
    2. Lead & Cap Suppression:
       Computes the column thickness profile (sum of mask pixels along each vertical column).
       Since wire leads are significantly thinner than the painted ceramic body, thresholding
       the thickness profile cleanly strips the leads, isolating the active body.
    3. Band Slicing (Vertical Sobel & 1D Windows):
       Applies vertical Sobel filtering (dx=1, dy=0) to the central 60% vertical strip
       to detect sharp chromatic/luminance transitions between colour bands. Slices individual
       band patches via 1D sliding windows and exports standardized 64x64 crops.
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ensure local imports work across environments
_curr_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
import sys
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

try:
    from stage2_segmentation import segment_apples_otsu, segment_resistors_adaptive
except ImportError:
    from Jakub.stage2_segmentation import segment_apples_otsu, segment_resistors_adaptive


# ==============================================================================
# TASK 1: APPLE MORPHOLOGY & WATERSHED SPLITTING
# ==============================================================================

def apply_apple_morphology(binary_mask):
    """Applies 3x3 morphological opening and topological hole-filling."""
    kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_3x3)

    # Topological hole-filling
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(opened)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return opened, filled


def split_apple_clusters_watershed(rgb_img, filled_mask, peak_thresh_ratio=0.30):
    """
    Performs Euclidean Distance Transform and Marker-Controlled Watershed
    to separate touching and clustered apples.
    """
    dist_transform = cv2.distanceTransform(filled_mask, cv2.DIST_L2, 5)

    max_dist = dist_transform.max()
    if max_dist > 0:
        _, sure_fg = cv2.threshold(
            dist_transform, peak_thresh_ratio * max_dist, 255, cv2.THRESH_BINARY
        )
    else:
        sure_fg = np.zeros_like(filled_mask)
    sure_fg = sure_fg.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sure_bg = cv2.dilate(filled_mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    watershed_img = rgb_img.copy()
    markers = cv2.watershed(watershed_img, markers)
    num_instances = max(0, num_markers - 1)
    return dist_transform, markers, num_instances


def extract_apple_crops(rgb_img, markers, min_area=50, target_size=(64, 64)):
    """Extracts 64x64 standardized apple proposals from watershed markers."""
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

        M = cv2.moments(inst_mask)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        x, y, w, h = cv2.boundingRect(inst_mask)
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

        crop_64 = cv2.resize(patch_rgb, target_size, interpolation=cv2.INTER_AREA)
        mask_64 = cv2.resize(patch_mask, target_size, interpolation=cv2.INTER_NEAREST)

        crops.append(crop_64)
        crop_masks.append(mask_64)
        metadata.append({
            "label": label,
            "centroid": (cx, cy),
            "bbox": (x, y, w, h),
            "area": area,
            "type": "apple",
        })

    return crops, crop_masks, metadata


# ==============================================================================
# TASK 2: RESISTOR POSE ALIGNMENT, LEAD STRIPPING & BAND SLICING
# ==============================================================================

def align_and_slice_resistors(rgb_img, resistor_blobs, body_mask, target_size=(64, 64)):
    """
    For each resistor contour:
      1. Rotates horizontally using cv2.minAreaRect and cv2.warpAffine.
      2. Strips thin metallic wire leads based on column thickness profile.
      3. Slices color bands via vertical Sobel filtering on central 60% and 1D sliding windows.
      4. Exports standardized 64x64 band crops.
    """
    h_img, w_img = rgb_img.shape[:2]
    all_band_crops = []
    all_band_masks = []
    metadata = []
    aligned_resistor_bodies = []

    for idx, c in enumerate(resistor_blobs):
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), angle = rect

        # Orient horizontally: longest dimension along width
        if rw < rh:
            rot_angle = angle - 90.0 if angle >= 0 else angle + 90.0
            length, thickness = rh, rw
        else:
            rot_angle = angle
            length, thickness = rw, rh

        if length < 60 or thickness < 12:
            continue

        # Affine rotation around contour centroid
        M = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
        rotated_img = cv2.warpAffine(
            rgb_img, M, (w_img, h_img), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        rotated_mask = cv2.warpAffine(
            body_mask, M, (w_img, h_img), flags=cv2.INTER_NEAREST
        )

        # Crop horizontally aligned bounding box
        x1 = int(max(0, cx - length / 2))
        x2 = int(min(w_img, cx + length / 2))
        y1 = int(max(0, cy - thickness))
        y2 = int(min(h_img, cy + thickness))

        crop_resistor = rotated_img[y1:y2, x1:x2]
        crop_mask = rotated_mask[y1:y2, x1:x2]

        if crop_resistor.size == 0 or crop_resistor.shape[1] < 40 or crop_resistor.shape[0] < 10:
            continue

        # 2. Strip Wire Leads by Column Thickness Profile
        # The cylindrical body is much thicker than the thin metallic wire leads
        col_thickness = np.sum(crop_mask > 0, axis=0)
        max_thick = col_thickness.max()
        if max_thick == 0:
            continue

        body_cols = np.where(col_thickness > 0.35 * max_thick)[0]
        if len(body_cols) > 30:
            bx1, bx2 = body_cols[0], body_cols[-1]
            body_crop = crop_resistor[:, bx1:bx2]
            body_crop_mask = crop_mask[:, bx1:bx2]
        else:
            body_crop = crop_resistor
            body_crop_mask = crop_mask

        aligned_resistor_bodies.append(body_crop)

        # 3. Band Slicing via Vertical Sobel & 1D Sliding Window
        bh, bw = body_crop.shape[:2]
        if bw < 30 or bh < 10:
            continue

        # Central 60% vertical crop (avoids top/bottom shadow edges and specular glare)
        center_strip = body_crop[int(bh * 0.2):int(bh * 0.8), :]
        gray_strip = cv2.cvtColor(center_strip, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray_strip, cv2.CV_64F, 1, 0, ksize=3)
        sobel_energy = np.mean(np.abs(sobel_x), axis=0)

        # 1D sliding window to extract standardized 64x64 band patches
        window_w = max(16, int(bw * 0.12))
        step = max(8, int(window_w * 0.8))

        band_idx = 0
        for wx in range(0, bw - window_w, step):
            band_patch = body_crop[:, wx:wx + window_w]
            band_mask_patch = body_crop_mask[:, wx:wx + window_w]

            band_64 = cv2.resize(band_patch, target_size, interpolation=cv2.INTER_AREA)
            bmask_64 = cv2.resize(band_mask_patch, target_size, interpolation=cv2.INTER_NEAREST)

            all_band_crops.append(band_64)
            all_band_masks.append(bmask_64)
            metadata.append({
                "resistor_id": idx,
                "band_index": band_idx,
                "centroid": (int(cx), int(cy)),
                "window_x": wx,
                "sobel_score": float(np.mean(sobel_energy[wx:wx + window_w])),
                "type": "resistor_band",
            })
            band_idx += 1

    return all_band_crops, all_band_masks, metadata, aligned_resistor_bodies


# ==============================================================================
# UNIFIED INTERFACE & EXECUTION
# ==============================================================================

def extract_proposals(rgb_img, mask, mode="apple"):
    """
    Unified Stage 3 entry point for both Apples and Resistors.
    Returns:
        crops: list of np.ndarray (64, 64, 3)
        masks: list of np.ndarray (64, 64)
        metadata: list of dicts
    """
    if mode == "apple":
        _, filled = apply_apple_morphology(mask)
        _, markers, _ = split_apple_clusters_watershed(rgb_img, filled)
        crops, masks, meta = extract_apple_crops(rgb_img, markers)
        return crops, masks, meta
    elif mode == "resistor":
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = [c for c in contours if cv2.contourArea(c) > 500]
        crops, masks, meta, _ = align_and_slice_resistors(rgb_img, blobs, mask)
        return crops, masks, meta
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'apple' or 'resistor'.")


def run_stage3():
    print("=" * 60)
    print("Machine Vision (UFMFRR-15-M) - Stage 3: Morphology & Proposals")
    print("=" * 60)

    os.makedirs("output", exist_ok=True)

    # ---------------------------------------------------------
    # 1. Process Apple Scenario (Watershed Splitting)
    # ---------------------------------------------------------
    apple_files = (
        glob.glob("minneapple-1-1/train/*.jpg")
        or glob.glob("/content/minneapple-1-1/train/*.jpg")
        or glob.glob("**/train/*.jpg", recursive=True)
    )
    if apple_files:
        apple_bgr = cv2.imread(apple_files[0])
        apple_rgb = cv2.cvtColor(apple_bgr, cv2.COLOR_BGR2RGB)
        raw_mask, _, _ = segment_apples_otsu(apple_rgb)
        opened, filled = apply_apple_morphology(raw_mask)
        dist_map, markers, n_inst = split_apple_clusters_watershed(apple_rgb, filled)
        apple_crops, apple_masks, apple_meta = extract_apple_crops(apple_rgb, markers)
        print(f"[APPLE] Watershed detected {n_inst} instances -> {len(apple_crops)} crops exported (64x64).")

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 3)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(apple_rgb)
        ax0.set_title(f"(a) Input Apple RGB: {os.path.basename(apple_files[0])}")
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[0, 1])
        im1 = ax1.imshow(dist_map, cmap="jet")
        ax1.set_title("(b) Euclidean Distance Transform (L2)")
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[0, 2])
        ws_vis = apple_rgb.copy()
        np.random.seed(42)
        colors = np.random.randint(0, 255, (max(markers.max() + 2, 10), 3), dtype=np.uint8)
        for m_id in np.unique(markers):
            if m_id >= 2:
                ws_vis[markers == m_id] = (0.5 * ws_vis[markers == m_id] + 0.5 * colors[m_id]).astype(np.uint8)
        ws_vis[markers == -1] = [255, 255, 0]
        ax2.imshow(ws_vis)
        ax2.set_title(f"(c) Watershed Instances ({len(apple_crops)} crops)")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[1, :])
        n_disp = min(len(apple_crops), 10)
        if n_disp > 0:
            strip = np.hstack([apple_crops[i] for i in range(n_disp)])
            ax3.imshow(strip)
            ax3.set_title(f"(d) 64x64 Normalized Proposal Crops (Showing {n_disp} of {len(apple_crops)})")
        ax3.axis("off")

        plt.tight_layout()
        out_apple = os.path.join("output", "stage3_apple_output.png")
        plt.savefig(out_apple, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[APPLE] Saved Stage 3 diagnostic: {out_apple}")

    # ---------------------------------------------------------
    # 2. Process Resistor Scenario (Pose, Lead Strip, Slicing)
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

        body_mask, _, blobs = segment_resistors_adaptive(res_rgb, min_area=500)
        band_crops, band_masks, band_meta, aligned_bodies = align_and_slice_resistors(
            res_rgb, blobs, body_mask
        )
        print(f"[RESISTOR] Aligned {len(aligned_bodies)} bodies -> {len(band_crops)} band crops exported (64x64).")

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(res_rgb)
        ax0.set_title(f"(a) Input Resistor RGB: {os.path.basename(resistor_files[0])}")
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[0, 1])
        if aligned_bodies:
            # Show the first horizontally aligned resistor body with leads stripped
            sample_body = aligned_bodies[0]
            ax1.imshow(sample_body)
            ax1.set_title(f"(b) Horizontally Aligned Body & Stripped Leads (Shape: {sample_body.shape})")
        else:
            ax1.text(0.5, 0.5, "No aligned bodies found", ha="center")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[1, :])
        n_disp = min(len(band_crops), 12)
        if n_disp > 0:
            band_strip = np.hstack([band_crops[i] for i in range(n_disp)])
            ax2.imshow(band_strip)
            ax2.set_title(f"(c) Sliced 64x64 Colour Band Proposals (Showing {n_disp} of {len(band_crops)})")
        else:
            ax2.text(0.5, 0.5, "No bands sliced", ha="center")
        ax2.axis("off")

        plt.tight_layout()
        out_res = os.path.join("output", "stage3_resistor_output.png")
        plt.savefig(out_res, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[RESISTOR] Saved Stage 3 diagnostic: {out_res}")

    print("\nStage 3 completed successfully across both tasks!")
    print("=" * 60)


if __name__ == "__main__":
    run_stage3()
