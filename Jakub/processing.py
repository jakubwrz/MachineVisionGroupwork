"""
Machine Vision (UFMFRR-15-M) - Group Portfolio Project
Part 1: Classical Image Processing Front-End for YOLO

Primary Contributor: Jakub Wrzyszcz
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# CONFIGURATION TOGGLE
# ==============================================================================
# False -> saves unenhanced crops to output/resistor_crops_raw/ (annotate these + baseline run)
# True  -> saves CLAHE/sharpened crops to output/resistor_crops_enhanced/ (enhanced ablation run)
ENHANCE_ON = False


def list_images(folder):
    """Returns a sorted list of all JPG/JPEG/PNG image paths in a folder."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(set(paths))


def clear_folder(folder):
    """Creates a folder if needed and clears existing image files inside it."""
    os.makedirs(folder, exist_ok=True)
    for old in list_images(folder):
        try:
            os.remove(old)
        except OSError:
            pass


def load_image(img_path, max_dim=1280):
    """Loads a BGR image (OpenCV & Ultralytics YOLO native format) and optionally caps max dimension."""
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    if max_dim is not None:
        h, w = bgr.shape[:2]
        if max(h, w) > max_dim:
            scale = float(max_dim) / float(max(h, w))
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return bgr


# ==============================================================================
# STAGE 1: SHARED PREPROCESSING (APPLES & RESISTORS)
# ==============================================================================

def preprocess_image(bgr_img, mode="apple"):
    """
    Normalizes lighting via LAB L-channel CLAHE and suppresses noise without shifting Hue.
    Returns a BGR image of the exact same dimensions, ready directly for YOLO.
    """
    clahe_clip = 2.5 if mode == "apple" else 2.0

    # 1. Equalize luminance (L) only so apple and resistor band colours stay accurate
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_eq = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(l_chan)
    eq_bgr = cv2.cvtColor(cv2.merge([l_eq, a_chan, b_chan]), cv2.COLOR_LAB2BGR)

    # 2. Light 3x3 median + Gaussian filtering removes specular/sensor noise without blurring bands
    denoised = cv2.medianBlur(eq_bgr, 3)
    clean_bgr = cv2.GaussianBlur(denoised, (3, 3), 0.8)
    return clean_bgr


# ==============================================================================
# STAGE 2: RESISTOR BODY ISOLATION, ALIGNMENT & BAND-CONTRAST ENHANCEMENT
# ==============================================================================

def enhance_resistor_crop(bgr_crop):
    """
    Applies local CLAHE, mild saturation boost (Hue untouched), and unsharp masking
    strictly inside the cropped resistor body so colour bands are sharp and distinct for YOLO.
    """
    # 1. Local CLAHE on luminance to pull dark bands (Brown/Red/Blue/Black) out of shadow
    lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(l_ch)
    contrast_bgr = cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # 2. Boost saturation by 25% without altering Hue
    hsv = cv2.cvtColor(contrast_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    s_boost = np.clip(s_ch.astype(np.float32) * 1.25, 0, 255).astype(np.uint8)
    vivid_bgr = cv2.cvtColor(cv2.merge([h_ch, s_boost, v_ch]), cv2.COLOR_HSV2BGR)

    # 3. Unsharp masking to sharpen vertical band edges
    blur = cv2.GaussianBlur(vivid_bgr, (0, 0), sigmaX=1.2)
    sharp_bgr = cv2.addWeighted(vivid_bgr, 1.4, blur, -0.4, 0)
    return sharp_bgr


def extract_resistor_bodies(bgr_img, enhance=ENHANCE_ON, raw_bgr=None):
    """
    Isolates each resistor's ceramic body, removes wire leads via Morphological Opening,
    rotates each body horizontally, and optionally enhances band contrast for YOLO.
    If enhance=False and raw_bgr is provided, crops are sliced from raw_bgr using the
    exact same mask/angle so pixel coordinates remain identical across both runs.
    """
    h_img, w_img = bgr_img.shape[:2]
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l_f, a_f, b_f = [c.astype(np.float32) for c in cv2.split(lab)]

    # 1. Estimate smooth local background with a wide effective window (~250px)
    #    so close-up resistor bodies are never absorbed into the median background
    small = cv2.resize(lab, (max(32, w_img // 8), max(32, h_img // 8)), interpolation=cv2.INTER_AREA)
    bg_lab = cv2.resize(cv2.medianBlur(small, 31), (w_img, h_img), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    # 2. Separate ceramic bodies from paper, soft wire shadows, and silver leads
    dL = np.maximum(bg_lab[:, :, 0] - l_f, 0.0)
    dC = np.sqrt((a_f - bg_lab[:, :, 1]) ** 2 + (b_f - bg_lab[:, :, 2]) ** 2)

    # Dark bodies (dL >= 60), vivid bodies (dC >= 14), or medium tan/olive bodies on warm desks
    raw_bin = (((dL >= 60.0) | (dC >= 14.0) | ((dL >= 35.0) & (dC >= 7.0))).astype(np.uint8)) * 255
    raw_bin = cv2.morphologyEx(raw_bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # 3. Amputate thin wire leads first, then bridge any narrow ceramic waist
    raw_cnts, _ = cv2.findContours(raw_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    body_mask = np.zeros_like(raw_bin)
    candidates = []

    for c in raw_cnts:
        if cv2.contourArea(c) < 350:
            continue
        comp = np.zeros_like(raw_bin)
        cv2.drawContours(comp, [c], -1, 255, thickness=cv2.FILLED)

        max_d = float(cv2.distanceTransform(comp, cv2.DIST_L2, 5).max())
        if max_d < 7.0:
            continue

        # Scale both lead-amputation opening and waist-bridging closing to this resistor's zoom level
        r_open = max(5, int(0.55 * max_d))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_open + 1, 2 * r_open + 1))
        k_bridge_sz = max(11, int(1.1 * max_d)) | 1
        k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_bridge_sz, k_bridge_sz))

        body = cv2.morphologyEx(cv2.morphologyEx(comp, cv2.MORPH_OPEN, k_open), cv2.MORPH_CLOSE, k_bridge)

        for bc in cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(bc) < 300:
                continue

            rect = cv2.minAreaRect(bc)
            box = cv2.boxPoints(rect)
            e1, e2 = box[1] - box[0], box[2] - box[1]
            l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
            body_len, body_thick = max(l1, l2), min(l1, l2)
            aspect = body_len / (body_thick + 1e-6)

            if body_len < 30 or body_thick < 14 or not (1.4 <= aspect <= 4.5):
                continue

            long_vec = e1 if l1 >= l2 else e2
            candidates.append((bc, rect, body_len, body_thick, long_vec))

    if not candidates:
        return [], [], body_mask

    # 4. Reject disconnected thin wire stubs (< 50% of the thickest ceramic body in the frame)
    max_thick = max(item[3] for item in candidates)
    src_for_crop = raw_bgr if (not enhance and raw_bgr is not None) else bgr_img
    crops, bboxes = [], []

    for bc, rect, body_len, body_thick, long_vec in candidates:
        if body_thick < 0.50 * max_thick:
            continue

        angle = float(np.degrees(np.arctan2(long_vec[1], long_vec[0])))
        angle = angle - 180.0 if angle > 90.0 else (angle + 180.0 if angle < -90.0 else angle)

        cx, cy = rect[0]
        M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rot_bgr = cv2.warpAffine(src_for_crop, M_rot, (w_img, h_img), borderMode=cv2.BORDER_REPLICATE)

        x1, x2 = int(max(0, cx - body_len * 0.58)), int(min(w_img, cx + body_len * 0.58))
        y1, y2 = int(max(0, cy - body_thick * 0.75)), int(min(h_img, cy + body_thick * 0.75))
        crop = rot_bgr[y1:y2, x1:x2]
        if crop.shape[0] >= 12 and crop.shape[1] >= 24:
            if enhance:
                crop = enhance_resistor_crop(crop)
            crops.append(crop)
            bboxes.append(list(cv2.boundingRect(bc)))
            cv2.drawContours(body_mask, [bc], -1, 255, thickness=cv2.FILLED)

    return crops, bboxes, body_mask


# ==============================================================================
# STAGE 3: UNIFIED YOLO OUTPUT COUNTER (APPLES & RESISTOR BANDS)
# ==============================================================================

def count_yolo_detections(yolo_result, mode="apple"):
    """
    Converts an Ultralytics YOLO result object into the counts required by the brief:
      - 'apple'    : returns total apple count.
      - 'resistor' : returns left-to-right band sequence and per-colour count dictionary.
    """
    boxes = yolo_result.boxes
    if mode == "apple":
        return {"total_apples": len(boxes)}

    detections = []
    for box in boxes:
        x_center = float(box.xywh[0][0])
        colour_name = yolo_result.names[int(box.cls[0])]
        detections.append((x_center, colour_name))
    detections.sort(key=lambda item: item[0])

    ordered_bands = [colour for _, colour in detections]
    colour_counts = {c: ordered_bands.count(c) for c in sorted(set(ordered_bands))}
    colour_counts["Total Bands"] = len(ordered_bands)
    return {"bands": ordered_bands, "colour_counts": colour_counts}


# ==============================================================================
# BATCH DATASET PROCESSING & DIAGNOSTIC VISUALISATION
# ==============================================================================

def save_resistor_debug_figure(clean_bgr, mask, bboxes, out_path):
    """Saves a 3-panel resistor overview (Stage 1 | Ceramic Mask | Isolated Detections)."""
    clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(clean_rgb); axes[0].set_title("(a) Stage 1 Preprocessed"); axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("(b) Stage 2 Ceramic Mask"); axes[1].axis("off")
    axes[2].imshow(clean_rgb); axes[2].set_title(f"(c) {len(bboxes)} Resistors Isolated for YOLO"); axes[2].axis("off")

    for i, (x, y, w, h) in enumerate(bboxes):
        axes[2].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="cyan", linewidth=2))
        axes[2].text(
            x, max(12, y - 5), f"R{i}", color="yellow", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, pad=1.5),
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def batch_process_apples(src_dir="minneapple-1-1/train", dst_dir="output/processed_apples"):
    """Preprocesses all orchard images at original resolution (keeps YOLO labels aligned)."""
    clear_folder(dst_dir)
    img_paths = list_images(src_dir)
    for path in img_paths:
        raw = load_image(path, max_dim=None)
        clean = preprocess_image(raw, mode="apple")
        cv2.imwrite(os.path.join(dst_dir, os.path.basename(path)), clean)
    print(f"[Apples]    Saved {len(img_paths)} preprocessed images -> {dst_dir}/")
    return len(img_paths)


def batch_process_resistors(
    src_dir="Resistor Dataset",
    processed_dir="output/processed_resistors",
    crops_dir=None,
    debug_dir="output/debug_resistors",
    enhance=ENHANCE_ON,
    save_debug=True,
):
    """
    Preprocesses all multi-resistor photos, saves full processed images, extracts every
    resistor body crop (raw or enhanced depending on ENHANCE_ON), and optionally saves debug figures.
    """
    if crops_dir is None:
        crops_dir = "output/resistor_crops_enhanced" if enhance else "output/resistor_crops_raw"

    clear_folder(processed_dir)
    clear_folder(crops_dir)
    if save_debug:
        clear_folder(debug_dir)

    img_paths = list_images(src_dir)
    total_crops = 0

    for path in img_paths:
        fname = os.path.basename(path)
        stem = os.path.splitext(fname)[0]

        raw = load_image(path, max_dim=1280)
        clean = preprocess_image(raw, mode="resistor")
        cv2.imwrite(os.path.join(processed_dir, fname), clean)

        crops, bboxes, mask = extract_resistor_bodies(clean, enhance=enhance, raw_bgr=raw)
        for i, crop in enumerate(crops):
            cv2.imwrite(os.path.join(crops_dir, f"{stem}_r{i:02d}.png"), crop)
            total_crops += 1

        if save_debug:
            save_resistor_debug_figure(clean, mask, bboxes, os.path.join(debug_dir, f"debug_{stem}.png"))

    mode_label = "enhanced" if enhance else "raw"
    print(f"[Resistors] Saved {len(img_paths)} preprocessed images -> {processed_dir}/")
    print(f"[Resistors] Exported {total_crops} {mode_label} horizontal crops -> {crops_dir}/")
    if save_debug:
        print(f"[Resistors] Saved {len(img_paths)} debug overlays -> {debug_dir}/")
    return len(img_paths), total_crops


if __name__ == "__main__":
    batch_process_apples(
        src_dir="minneapple-1-1/train",
        dst_dir="output/processed_apples",
    )
    batch_process_resistors(
        src_dir="Resistor Dataset",
        processed_dir="output/processed_resistors",
        debug_dir="output/debug_resistors",
        enhance=ENHANCE_ON,
        save_debug=True,
    )