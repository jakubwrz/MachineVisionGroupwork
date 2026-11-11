"""
Machine Vision (UFMFRR-15-M) - Group Project
Stage 1: Radiometric & Spatial Preprocessing (Units 2, 4, 5)

Primary Contributor: [Name]

Unified Self-Contained Execution Script for Google Colab & Local Environments.
Processes both:
  - Task 1: MinneApple Orchard Dataset (Real COCO Annotations + 512x512 Tiling)
  - Task 2: Multi-Resistor Colour Bands (Specular Reflection Glare Suppression)
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Generator
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ==============================================================================
# SECTION 1: UNIT 2 - COLOUR SPACE TRANSFORMATIONS (GONZALEZ & WOODS)
# ==============================================================================

def rgb_to_hsi(rgb_img: np.ndarray) -> np.ndarray:
    """
    Converts RGB image [0, 255] or [0.0, 1.0] to HSI colour space
    using exact Gonzalez & Woods equations (Unit 2).
    """
    if rgb_img.dtype == np.uint8:
        rgb = rgb_img.astype(np.float32) / 255.0
    else:
        rgb = rgb_img.astype(np.float32)
        if rgb.max() > 1.0:
            rgb /= 255.0

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    eps = 1e-7

    # 1. Intensity (I)
    intensity = (r + g + b) / 3.0

    # 2. Saturation (S)
    rgb_sum = r + g + b
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = np.zeros_like(intensity)
    non_zero = rgb_sum > eps
    saturation[non_zero] = 1.0 - (3.0 / rgb_sum[non_zero]) * min_rgb[non_zero]
    saturation = np.clip(saturation, 0.0, 1.0)

    # 3. Hue (H)
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + eps
    arg = np.clip(num / den, -1.0, 1.0)
    theta = np.arccos(arg)

    hue = np.zeros_like(intensity)
    b_greater_g = b > g
    hue[~b_greater_g] = theta[~b_greater_g]
    hue[b_greater_g] = (2.0 * np.pi) - theta[b_greater_g]
    hue = hue / (2.0 * np.pi)  # Normalise to [0.0, 1.0]
    hue[saturation < eps] = 0.0  # Undefined for achromatic pixels

    return np.stack([hue, saturation, intensity], axis=-1).astype(np.float32)


def hsi_to_rgb(hsi_img: np.ndarray) -> np.ndarray:
    """
    Converts HSI image [0.0, 1.0] back to RGB uint8 [0, 255]
    using sectoral reconstruction (Unit 2).
    """
    h = hsi_img[:, :, 0] * (2.0 * np.pi)
    s = np.clip(hsi_img[:, :, 1], 0.0, 1.0)
    i = np.clip(hsi_img[:, :, 2], 0.0, 1.0)

    r = np.zeros_like(i)
    g = np.zeros_like(i)
    b = np.zeros_like(i)

    rad_120 = 2.0 * np.pi / 3.0
    rad_240 = 4.0 * np.pi / 3.0

    # Sector 1: 0 <= H < 120 deg
    sec1 = (h >= 0.0) & (h < rad_120)
    cos_denom1 = np.where(np.abs(np.cos(np.pi / 3.0 - h[sec1])) < 1e-7, 1e-7, np.cos(np.pi / 3.0 - h[sec1]))
    b[sec1] = i[sec1] * (1.0 - s[sec1])
    r[sec1] = i[sec1] * (1.0 + (s[sec1] * np.cos(h[sec1])) / cos_denom1)
    g[sec1] = 3.0 * i[sec1] - (r[sec1] + b[sec1])

    # Sector 2: 120 <= H < 240 deg
    sec2 = (h >= rad_120) & (h < rad_240)
    h_sec2 = h[sec2] - rad_120
    cos_denom2 = np.where(np.abs(np.cos(np.pi / 3.0 - h_sec2)) < 1e-7, 1e-7, np.cos(np.pi / 3.0 - h_sec2))
    r[sec2] = i[sec2] * (1.0 - s[sec2])
    g[sec2] = i[sec2] * (1.0 + (s[sec2] * np.cos(h_sec2)) / cos_denom2)
    b[sec2] = 3.0 * i[sec2] - (r[sec2] + g[sec2])

    # Sector 3: 240 <= H < 360 deg
    sec3 = (h >= rad_240)
    h_sec3 = h[sec3] - rad_240
    cos_denom3 = np.where(np.abs(np.cos(np.pi / 3.0 - h_sec3)) < 1e-7, 1e-7, np.cos(np.pi / 3.0 - h_sec3))
    g[sec3] = i[sec3] * (1.0 - s[sec3])
    b[sec3] = i[sec3] * (1.0 + (s[sec3] * np.cos(h_sec3)) / cos_denom3)
    r[sec3] = 3.0 * i[sec3] - (g[sec3] + b[sec3])

    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


# ==============================================================================
# SECTION 2: UNIT 4 - CONTRAST NORMALISATION (CDF HISTOGRAM EQUALISATION & CLAHE)
# ==============================================================================

def histogram_equalization_cdf(channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs Global Histogram Equalisation from first principles using CDF mapping (Unit 4).
    """
    if channel.dtype != np.uint8:
        channel = np.clip(channel, 0, 255).astype(np.uint8)

    hist, _ = np.histogram(channel.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_max = cdf[-1]
    cdf_normalized = cdf / float(cdf_max)

    # Masked CDF to stretch active dynamic range
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_min = cdf_m.min()
    if cdf_max > cdf_min:
        lut = (cdf_m - cdf_min) * 255.0 / (cdf_max - cdf_min)
    else:
        lut = cdf_m * 0

    lut = np.ma.filled(lut, 0).round().astype(np.uint8)
    equalized_channel = lut[channel]
    return equalized_channel, hist, cdf_normalized


def apply_clahe(channel: np.ndarray,
                clip_limit: float = 2.5,
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalisation (CLAHE) (Unit 4).
    """
    if channel.dtype != np.uint8:
        channel = np.clip(channel, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(channel)


# ==============================================================================
# SECTION 3: UNIT 5 - SPATIAL DENOISING (MEDIAN & GAUSSIAN FILTERING)
# ==============================================================================

def apply_spatial_denoising(channel: np.ndarray,
                           median_ksize: int = 3,
                           gaussian_ksize: int = 3,
                           gaussian_sigma: float = 0.8) -> np.ndarray:
    """
    Applies non-linear Median Filtering followed by linear Gaussian Smoothing (Unit 5).
    - Median Filter: Suppresses impulse noise & thin specular reflection glints without edge blurring.
    - Gaussian Filter: Smooths high-frequency sensor noise.
    """
    out = channel.copy()
    if median_ksize > 1:
        if median_ksize % 2 == 0:
            median_ksize += 1
        out = cv2.medianBlur(out, median_ksize)

    if gaussian_ksize > 1:
        if gaussian_ksize % 2 == 0:
            gaussian_ksize += 1
        out = cv2.GaussianBlur(out, (gaussian_ksize, gaussian_ksize), gaussian_sigma)

    return out


# ==============================================================================
# SECTION 4: UNIFIED CONFIGURATIONS & PREPROCESSOR CLASS
# ==============================================================================

DEFAULT_APPLE_CONFIG: Dict[str, Any] = {
    "colour_space": "hsi",
    "equalisation_method": "clahe",
    "clahe_clip_limit": 2.5,
    "clahe_tile_grid_size": (8, 8),
    "median_ksize": 3,
    "gaussian_ksize": 3,
    "gaussian_sigma": 0.8,
    "denoise_target": "intensity",
}

DEFAULT_RESISTOR_CONFIG: Dict[str, Any] = {
    "colour_space": "hsi",
    "equalisation_method": "clahe",
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": (8, 8),
    "median_ksize": 5,             # 5x5 median eliminates specular reflection streaks on cylindrical body
    "gaussian_ksize": 3,
    "gaussian_sigma": 1.0,
    "denoise_target": "intensity",
}


class Stage1Preprocessor:
    def __init__(self, config: Optional[Dict[str, Any]] = None, mode: str = "apple"):
        if config is None:
            self.config = DEFAULT_APPLE_CONFIG.copy() if mode == "apple" else DEFAULT_RESISTOR_CONFIG.copy()
        else:
            self.config = config.copy()

    def process(self, rgb_img: np.ndarray) -> Dict[str, Any]:
        if rgb_img.dtype != np.uint8:
            rgb_uint8 = np.clip(rgb_img, 0, 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb_img.copy()

        cspace = self.config.get("colour_space", "hsi").lower()
        eq_method = self.config.get("equalisation_method", "clahe").lower()
        clip_limit = self.config.get("clahe_clip_limit", 2.5)
        tile_grid = tuple(self.config.get("clahe_tile_grid_size", (8, 8)))
        med_k = self.config.get("median_ksize", 3)
        gauss_k = self.config.get("gaussian_ksize", 3)
        gauss_s = self.config.get("gaussian_sigma", 0.8)

        # 1. Colour Space Conversion
        if cspace == "hsi":
            hsi = rgb_to_hsi(rgb_uint8)
            h = hsi[:, :, 0]
            s = hsi[:, :, 1]
            i_float = hsi[:, :, 2]
            ch3_raw_u8 = (i_float * 255.0).round().astype(np.uint8)
            ch1_disp = (h * 255.0).round().astype(np.uint8)
            ch2_disp = (s * 255.0).round().astype(np.uint8)
        elif cspace == "hsv":
            hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
            ch1_disp, ch2_disp, ch3_raw_u8 = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        elif cspace == "lab":
            lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
            ch3_raw_u8, ch1_disp, ch2_disp = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        raw_eq, raw_hist, raw_cdf = histogram_equalization_cdf(ch3_raw_u8)

        # 2. Contrast Normalisation on Intensity ONLY
        if eq_method == "clahe":
            ch3_eq = apply_clahe(ch3_raw_u8, clip_limit=clip_limit, tile_grid_size=tile_grid)
        elif eq_method == "global_he":
            ch3_eq = raw_eq
        else:
            ch3_eq = ch3_raw_u8.copy()

        # 3. Spatial Denoising
        ch3_denoised = apply_spatial_denoising(
            ch3_eq,
            median_ksize=med_k,
            gaussian_ksize=gauss_k,
            gaussian_sigma=gauss_s
        )

        eq_final, eq_hist, eq_cdf = histogram_equalization_cdf(ch3_denoised)

        # 4. Re-synthesise RGB
        if cspace == "hsi":
            hsi_processed = np.stack([h, s, ch3_denoised.astype(np.float32) / 255.0], axis=-1)
            preprocessed_rgb = hsi_to_rgb(hsi_processed)
        elif cspace == "hsv":
            hsv_processed = np.stack([ch1_disp, ch2_disp, ch3_denoised], axis=-1)
            preprocessed_rgb = cv2.cvtColor(hsv_processed, cv2.COLOR_HSV2RGB)
        elif cspace == "lab":
            lab_processed = np.stack([ch3_denoised, ch1_disp, ch2_disp], axis=-1)
            preprocessed_rgb = cv2.cvtColor(lab_processed, cv2.COLOR_LAB2RGB)

        return {
            "original_rgb": rgb_uint8,
            "preprocessed_rgb": preprocessed_rgb,
            "colour_space": cspace,
            "channel_1": ch1_disp,
            "channel_2": ch2_disp,
            "channel_3_raw": ch3_raw_u8,
            "channel_3_eq": ch3_eq,
            "channel_3_denoised": ch3_denoised,
            "raw_hist": raw_hist,
            "raw_cdf": raw_cdf,
            "eq_hist": eq_hist,
            "eq_cdf": eq_cdf,
            "config": self.config
        }


def preprocess_image(rgb_img: np.ndarray,
                     mode: str = "apple",
                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Unified entry point for Stage 1 preprocessing."""
    return Stage1Preprocessor(config=config, mode=mode).process(rgb_img)


# ==============================================================================
# SECTION 5: MINNEAPPLE DATASET LOADER & HIGH-RES TILING
# ==============================================================================

class MinneAppleDataset:
    """COCO-format dataset loader for MinneApple orchard images."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.annotation_path = os.path.join(self.dataset_dir, "_annotations.coco.json")
        if not os.path.exists(self.annotation_path):
            raise FileNotFoundError(f"Annotations not found at: {self.annotation_path}")

        with open(self.annotation_path, "r", encoding="utf-8") as f:
            self.coco_data = json.load(f)

        self.images = {img["id"]: img for img in self.coco_data.get("images", [])}
        self.image_ids = list(self.images.keys())
        self.annotations_by_image = {img_id: [] for img_id in self.image_ids}
        for ann in self.coco_data.get("annotations", []):
            if ann["image_id"] in self.annotations_by_image:
                self.annotations_by_image[ann["image_id"]].append(ann)

        print(f"Loaded MinneApple Dataset: {len(self.image_ids)} images, {len(self.coco_data.get('annotations', []))} annotations.")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        img_id = self.image_ids[idx]
        meta = self.images[img_id].copy()
        file_path = os.path.join(self.dataset_dir, meta["file_name"])
        bgr = cv2.imread(file_path)
        if bgr is None:
            raise ValueError(f"Failed to load image: {file_path}")
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        anns = self.annotations_by_image.get(img_id, [])
        meta["apple_count"] = len(anns)
        return img_rgb, anns, meta

    @staticmethod
    def tile_image(image: np.ndarray,
                   patch_size: int = 512,
                   overlap: float = 0.2) -> Generator[Tuple[np.ndarray, Tuple[int, int, int, int]], None, None]:
        """Slices high-res image into overlapping 512x512 patches with 20% stride overlap."""
        h, w = image.shape[:2]
        stride = int(patch_size * (1.0 - overlap))
        for y in range(0, max(1, h - patch_size + 1), stride):
            for x in range(0, max(1, w - patch_size + 1), stride):
                patch = image[y:y + patch_size, x:x + patch_size]
                yield patch, (y, y + patch_size, x, x + patch_size)


# ==============================================================================
# SECTION 6: DIAGNOSTIC VISUALISATION TOOLS
# ==============================================================================

def plot_stage1_diagnostics(result: Dict[str, Any],
                            title: str = "Stage 1 Preprocessing Diagnostics",
                            save_path: Optional[str] = None):
    """Generates 8-panel diagnostic figure comparing raw vs preprocessed stages & CDF."""
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(result["original_rgb"])
    ax1.set_title("1. Original RGB", fontsize=11, fontweight="semibold")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(result["preprocessed_rgb"])
    ax2.set_title("2. Preprocessed RGB", fontsize=11, fontweight="semibold")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 4, 3)
    im3 = ax3.imshow(result["channel_1"], cmap="hsv" if result["colour_space"] in ("hsi", "hsv") else "coolwarm")
    ax3.set_title(f"3. {result['colour_space'].upper()} Ch 1 (Hue/A)", fontsize=11, fontweight="semibold")
    ax3.axis("off")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, 4, 4)
    im4 = ax4.imshow(result["channel_2"], cmap="plasma")
    ax4.set_title(f"4. {result['colour_space'].upper()} Ch 2 (Sat/B)", fontsize=11, fontweight="semibold")
    ax4.axis("off")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(2, 4, 5)
    im5 = ax5.imshow(result["channel_3_raw"], cmap="gray")
    ax5.set_title("5. Raw Intensity Channel", fontsize=11, fontweight="semibold")
    ax5.axis("off")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(2, 4, 6)
    im6 = ax6.imshow(result["channel_3_denoised"], cmap="gray")
    ax6.set_title("6. CLAHE + Denoised Intensity", fontsize=11, fontweight="semibold")
    ax6.axis("off")
    fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    # 7. Raw Histogram & CDF
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.bar(range(256), result["raw_hist"], color="steelblue", alpha=0.6, width=1.0)
    ax7.set_xlim([0, 255])
    ax7.set_xlabel("Intensity Level $r_k$", fontsize=9)
    ax7.set_ylabel("Pixel Frequency", fontsize=9, color="steelblue")
    ax7_twin = ax7.twinx()
    ax7_twin.plot(range(256), result["raw_cdf"], color="darkred", linewidth=2.0)
    ax7_twin.set_ylim([0.0, 1.05])
    ax7_twin.set_ylabel("Normalised CDF", fontsize=9, color="darkred")
    ax7.set_title("7. Raw Histogram & CDF", fontsize=11, fontweight="semibold")

    # 8. Equalised Histogram & CDF
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.bar(range(256), result["eq_hist"], color="forestgreen", alpha=0.6, width=1.0)
    ax8.set_xlim([0, 255])
    ax8.set_xlabel("Intensity Level $s_k$", fontsize=9)
    ax8.set_ylabel("Pixel Frequency", fontsize=9, color="forestgreen")
    ax8_twin = ax8.twinx()
    ax8_twin.plot(range(256), result["eq_cdf"], color="darkmagenta", linewidth=2.0)
    ax8_twin.set_ylim([0.0, 1.05])
    ax8_twin.set_ylabel("Linearised CDF", fontsize=9, color="darkmagenta")
    ax8.set_title("8. Equalised Histogram & CDF", fontsize=11, fontweight="semibold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_specular_line_profile(raw_intensity: np.ndarray,
                               denoised_intensity: np.ndarray,
                               row_idx: int,
                               title: str = "Specular Reflection Glint Suppression (Unit 5)",
                               save_path: Optional[str] = None):
    """1D cross-sectional profile demonstrating specular spike suppression without edge blurring."""
    plt.figure(figsize=(11, 4.5))
    x = np.arange(raw_intensity.shape[1])
    plt.plot(x, raw_intensity[row_idx, :], color="crimson", linestyle="--", alpha=0.85, label="Raw Intensity (Specular Glare Spikes)")
    plt.plot(x, denoised_intensity[row_idx, :], color="navy", linewidth=2.0, label="Median Denoised (Specular Glint Suppressed)")
    plt.title(f"{title} (Row {row_idx})", fontsize=12, fontweight="bold")
    plt.xlabel("Horizontal Pixel Coordinate (x)", fontsize=10)
    plt.ylabel("Intensity Level (0-255)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ==============================================================================
# SECTION 7: MAIN EXECUTION ENTRY POINT
# ==============================================================================

def main():
    print("=" * 75)
    print("Machine Vision (UFMFRR-15-M) - Stage 1 Preprocessing Execution")
    print("=" * 75)

    # 1. Locate the MinneApple dataset
    candidate_dirs = [
        "minneapple-1-1/train",
        "/content/minneapple-1-1/train",
        "/content/MachineVisionGroupwork/minneapple-1-1/train",
        "/content/drive/MyDrive/minneapple-1-1/train",
        "../minneapple-1-1/train"
    ]
    
    dataset_dir = None
    for p in candidate_dirs:
        if os.path.exists(os.path.join(p, "_annotations.coco.json")):
            dataset_dir = p
            break

    if dataset_dir:
        print(f"\n[TASK 1] Found MinneApple Dataset at: {dataset_dir}")
        dataset = MinneAppleDataset(dataset_dir)

        # Process first sample image
        sample_img, anns, meta = dataset[0]
        print(f"  Processing Image: {meta['file_name']} ({meta['apple_count']} apples, shape {sample_img.shape})...")

        t0 = time.perf_counter()
        apple_res = preprocess_image(sample_img, mode="apple")
        t_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  -> Stage 1 Completed in {t_ms:.2f} ms")

        # Display Diagnostic Visualisation
        plot_stage1_diagnostics(
            apple_res,
            title=f"Task 1: MinneApple Preprocessing Diagnostics ({meta['apple_count']} Apples)"
        )

        # Demonstrate 512x512 Tiling (20% overlap)
        tiles = list(MinneAppleDataset.tile_image(sample_img, patch_size=512, overlap=0.2))
        print(f"  -> Extracted {len(tiles)} overlapping 512x512 patches covering canvas.")
    else:
        print("\n[NOTE] MinneApple dataset folder not found in default paths.")
        print("Please set your dataset path or pass it directly.")

    # 2. Multi-Resistor Benchmark (Task 2)
    print("\n[TASK 2] Demonstrating Multi-Resistor Specular Glare Preprocessing...")
    # Synthetic cylindrical resistor scene with glare streaks
    bg = np.random.normal(160, 8, (350, 600)).clip(130, 190).astype(np.uint8)
    res_img = np.stack([bg, bg, (bg * 0.95).astype(np.uint8)], axis=-1)
    cv2.line(res_img, (50, 120), (550, 120), (130, 130, 135), 4)
    cv2.rectangle(res_img, (180, 85), (420, 155), (170, 190, 210), -1)
    res_img[85:156, 215:227] = (120, 60, 20)  # Brown band
    res_img[85:156, 245:257] = (20, 20, 20)    # Black band
    res_img[85:156, 275:287] = (210, 25, 20)   # Red band
    res_img[85:156, 370:385] = (210, 175, 45)  # Gold band
    # Specular glare streak along row 112
    res_img[112:114, 190:420] = 255

    resistor_res = preprocess_image(res_img, mode="resistor")
    plot_stage1_diagnostics(resistor_res, title="Task 2: Resistor Specular Glare Preprocessing Diagnostics")
    plot_specular_line_profile(
        resistor_res["channel_3_raw"],
        resistor_res["channel_3_denoised"],
        row_idx=112,
        title="Specular Reflection Glint Suppression on Cylindrical Resistor (Row 112)"
    )

    print("=" * 75)
    print("Stage 1 execution complete!")
    print("=" * 75)


if __name__ == "__main__":
    main()
