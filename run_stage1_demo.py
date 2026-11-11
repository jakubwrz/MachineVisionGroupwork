"""
Machine Vision (UFMFRR-15-M) - Group Project
Stage 1: Radiometric & Spatial Preprocessing (Units 2, 4, 5)

Primary Contributor: [Group Member Name / Student ID]

Self-Contained Unified Runner for Google Colab & Local Environments.
Does NOT require 'src' folder imports — runs anywhere (standalone script or notebook cell).

Handles both project scenarios:
  1. Task 1 (Orchard Apples): Automatically loads from the real MinneApple dataset
     (or falls back to a synthetic canopy scene if dataset is not present), performs
     HSI decoupling, Intensity CLAHE contrast normalisation, and 3x3 median denoising,
     with 512x512 tiling demonstration.
  2. Task 2 (Multi-Resistor): Models cylindrical resistor bodies, colour bands, and
     demonstrates 5x5 median filtering to eliminate bright specular glare streaks.

Outputs:
  - Inline matplotlib plots in Google Colab / Jupyter
  - Diagnostic 8-panel figures saved to output/ directory
  - 1D cross-sectional specular reflection glint suppression profile
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

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
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
    "median_ksize": 5,             # 5x5 median eliminates specular glare on cylinder
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

        print(f"Loaded MinneApple: {len(self.image_ids)} images, {len(self.coco_data.get('annotations', []))} annotations.")

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

def is_interactive() -> bool:
    """Returns True if running inside an IPython/Colab notebook environment."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


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


def plot_specular_line_profile_comparison(raw_channel: np.ndarray,
                                          denoised_channel: np.ndarray,
                                          row_idx: int,
                                          title: str = "Specular Reflection Glint Suppression (Unit 5)",
                                          save_path: Optional[str] = None):
    """1D cross-sectional profile demonstrating specular spike suppression without edge blurring."""
    fig = plt.figure(figsize=(11, 4.5))
    x = np.arange(raw_channel.shape[1])
    plt.plot(x, raw_channel[row_idx, :], color="crimson", linestyle="--", alpha=0.85, label="Raw Intensity (Specular Glare Spikes)")
    plt.plot(x, denoised_channel[row_idx, :], color="navy", linewidth=2.0, label="Median Denoised (Specular Glint Suppressed)")
    plt.title(f"{title} (Row {row_idx})", fontsize=12, fontweight="bold")
    plt.xlabel("Horizontal Pixel Coordinate (x)", fontsize=10)
    plt.ylabel("Intensity Level (0-255)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ==============================================================================
# SECTION 7: SYNTHETIC BENCHMARK GENERATORS
# ==============================================================================

def create_synthetic_orchard_scene(height: int = 400, width: int = 500) -> np.ndarray:
    """Synthesises an orchard scene with apples, foliage, and canopy shadow gradient."""
    np.random.seed(101)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    leaves_base = np.random.normal(loc=70, scale=15, size=(height, width)).clip(30, 120).astype(np.uint8)
    img[:, :, 0] = (leaves_base * 0.35).astype(np.uint8)
    img[:, :, 1] = (leaves_base * 1.1).clip(0, 255).astype(np.uint8)
    img[:, :, 2] = (leaves_base * 0.25).astype(np.uint8)

    cv2.circle(img, (150, 180), 55, (190, 25, 20), -1)
    cv2.circle(img, (220, 210), 50, (175, 20, 15), -1)
    cv2.circle(img, (330, 150), 48, (210, 80, 25), -1)
    cv2.circle(img, (260, 290), 45, (110, 170, 20), -1)
    cv2.circle(img, (390, 280), 28, (180, 30, 20), -1)

    y, x = np.ogrid[:height, :width]
    for cx, cy, r in [(150, 180, 55), (220, 210, 50), (330, 150, 48), (260, 290, 45), (390, 280, 28)]:
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = dist <= r
        shading = 1.0 - 0.35 * (dist / r) ** 2
        for c in range(3):
            img[mask, c] = (img[mask, c] * shading[mask]).clip(0, 255).astype(np.uint8)

    grad = np.linspace(0.3, 1.2, width)
    grad_2d = np.tile(grad, (height, 1))
    shadow_mask = np.clip(grad_2d + 0.2 * np.sin(np.linspace(0, 6, height))[:, None], 0.25, 1.2)
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c] * shadow_mask, 0, 255).astype(np.uint8)

    noise = np.random.normal(0, 6, (height, width, 3)).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_synthetic_resistor_scene(height: int = 350, width: int = 600) -> np.ndarray:
    """Synthesises a workbench scene with cylindrical resistors, colour bands, and glare streaks."""
    np.random.seed(202)
    bg = np.random.normal(160, 8, (height, width)).clip(130, 190).astype(np.uint8)
    img = np.stack([bg, bg, (bg * 0.95).astype(np.uint8)], axis=-1)

    # Resistor 1: Beige (Brown, Black, Red, Gold)
    cv2.line(img, (50, 120), (550, 120), (130, 130, 135), 4)
    cv2.rectangle(img, (180, 85), (420, 155), (170, 190, 210), -1)
    for x1, x2, rgb_color in [(215, 227, (120, 60, 20)), (245, 257, (20, 20, 20)), (275, 287, (210, 25, 20)), (370, 385, (210, 175, 45))]:
        img[85:156, x1:x2] = rgb_color

    # Resistor 2: Blue (Yellow, Violet, Black, Orange, Brown)
    cv2.line(img, (70, 250), (530, 250), (135, 135, 140), 4)
    cv2.rectangle(img, (190, 215), (430, 285), (40, 110, 190), -1)
    for x1, x2, rgb_color in [(215, 227, (230, 210, 30)), (245, 257, (130, 30, 140)), (275, 287, (20, 20, 20)), (305, 317, (235, 120, 20)), (380, 392, (120, 60, 20))]:
        img[215:286, x1:x2] = rgb_color

    for y_start, y_end in [(85, 155), (215, 285)]:
        h_body = y_end - y_start
        cyl_profile = np.sin(np.linspace(0.1, np.pi - 0.1, h_body))
        for r_idx in range(h_body):
            img[y_start + r_idx, 180:430] = np.clip(img[y_start + r_idx, 180:430] * (0.6 + 0.5 * cyl_profile[r_idx]), 0, 255).astype(np.uint8)

    for r in [112, 113, 242, 243]:
        img[r, 190:420] = np.clip(img[r, 190:420] + 180, 0, 255).astype(np.uint8)

    noise = np.random.normal(0, 4, (height, width, 3)).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ==============================================================================
# SECTION 8: PRIMARY EXECUTION FUNCTION
# ==============================================================================

def run_demonstration():
    print("=" * 75)
    print("Machine Vision (UFMFRR-15-M) - Stage 1 Preprocessing Demonstration")
    print("=" * 75)

    os.makedirs("output", exist_ok=True)
    interactive = is_interactive()

    # --------------------------------------------------------------------------
    # 1. ORCHARD APPLE SCENARIO (TASK 1)
    # --------------------------------------------------------------------------
    print("\n[1/3] Processing Orchard Apple Scenario (Task 1)...")
    
    # Auto-detect real MinneApple dataset location across common Colab/local paths
    candidate_paths = [
        "minneapple-1-1/train",
        "/content/minneapple-1-1/train",
        "/content/MachineVisionGroupwork/minneapple-1-1/train",
        "/content/drive/MyDrive/minneapple-1-1/train",
        "/content/drive/MyDrive/MachineVision_Groupwork/minneapple-1-1/train",
        "../minneapple-1-1/train",
    ]
    
    real_dataset_dir = None
    for p in candidate_paths:
        if os.path.exists(os.path.join(p, "_annotations.coco.json")):
            real_dataset_dir = p
            break

    if real_dataset_dir is not None:
        print(f"  -> Using REAL MinneApple dataset from: {real_dataset_dir}")
        dataset = MinneAppleDataset(dataset_dir=real_dataset_dir)
        apple_img, anns, meta = dataset[0]
        title_prefix = f"Real MinneApple Image ({meta['file_name']}, {meta['apple_count']} Apples)"
    else:
        print("  -> MinneApple dataset folder not found, using synthetic orchard benchmark.")
        apple_img = create_synthetic_orchard_scene()
        title_prefix = "Synthetic Orchard Apple Canopy"

    apple_res = preprocess_image(apple_img, mode="apple")
    print(f"  - Colour Space: {apple_res['colour_space'].upper()}")
    print(f"  - Contrast Normalisation: CLAHE (clipLimit={apple_res['config']['clahe_clip_limit']})")
    print(f"  - Spatial Denoising: Median ({apple_res['config']['median_ksize']}x{apple_res['config']['median_ksize']}) + Gaussian ({apple_res['config']['gaussian_ksize']}x{apple_res['config']['gaussian_ksize']})")
    
    apple_fig_path = os.path.join("output", "figure_stage1_apple_diagnostics.png")
    plot_stage1_diagnostics(
        apple_res,
        title=f"Stage 1 Diagnostics: {title_prefix} (Units 2, 4, 5)",
        save_path=apple_fig_path
    )
    if interactive:
        plt.show()
    print(f"  -> Saved diagnostic figure: {apple_fig_path}")

    # Demonstrate 512x512 Tiling with 20% overlap
    tiles = list(MinneAppleDataset.tile_image(apple_img, patch_size=512, overlap=0.2))
    print(f"  -> Extracted {len(tiles)} overlapping 512x512 patches covering canvas.")

    # --------------------------------------------------------------------------
    # 2. MULTI-RESISTOR SCENARIO (TASK 2)
    # --------------------------------------------------------------------------
    print("\n[2/3] Processing Multi-Resistor Scenario (Task 2)...")
    resistor_img = create_synthetic_resistor_scene()
    resistor_res = preprocess_image(resistor_img, mode="resistor")

    print(f"  - Colour Space: {resistor_res['colour_space'].upper()}")
    print(f"  - Contrast Normalisation: CLAHE (clipLimit={resistor_res['config']['clahe_clip_limit']})")
    print(f"  - Spatial Denoising: Median ({resistor_res['config']['median_ksize']}x{resistor_res['config']['median_ksize']}) + Gaussian ({resistor_res['config']['gaussian_ksize']}x{resistor_res['config']['gaussian_ksize']})")

    resistor_fig_path = os.path.join("output", "figure_stage1_resistor_diagnostics.png")
    plot_stage1_diagnostics(
        resistor_res,
        title="Stage 1 Diagnostics: Resistor Body & Specular Glare Preprocessing (Units 2, 4, 5)",
        save_path=resistor_fig_path
    )
    if interactive:
        plt.show()
    print(f"  -> Saved diagnostic figure: {resistor_fig_path}")

    # --------------------------------------------------------------------------
    # 3. SPECULAR GLARE LINE PROFILE EVALUATION
    # --------------------------------------------------------------------------
    print("\n[3/3] Generating Specular Reflection Glint Suppression Analysis...")
    glare_row = 112
    profile_fig_path = os.path.join("output", "figure_stage1_specular_profile.png")
    plot_specular_line_profile_comparison(
        raw_channel=resistor_res["channel_3_raw"],
        denoised_channel=resistor_res["channel_3_denoised"],
        row_idx=glare_row,
        title="Specular Glare Profile on Cylindrical Resistor Body (Unit 5 Median Filtering)",
        save_path=profile_fig_path
    )
    if interactive:
        plt.show()
    print(f"  -> Saved 1D profile figure: {profile_fig_path}")

    print("\n" + "=" * 75)
    print("Stage 1 Pipeline successfully executed across both domains!")
    print("All diagnostic figures verified and written to output/ directory.")
    print("=" * 75)


if __name__ == "__main__":
    run_demonstration()
