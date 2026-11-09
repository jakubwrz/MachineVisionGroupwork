"""
Machine Vision (UFMFRR-15-M) - Group Project
Stage 1: Radiometric & Spatial Preprocessing Module (Units 2, 4, 5)

Primary Contributor: [Group Member Name / Student ID]
Shared Unified Architecture for Task 1 (Orchard Apples) & Task 2 (Resistor Colour Bands)

Technical Overview:
-------------------
Raw RGB imaging entangles intrinsic surface reflectance (chrominance) with extrinsic
illumination geometry and intensity (luminance). 
- In Task 1 (MinneApple), outdoor canopy foliage suffers from harsh solar shadowing,
  dappled lighting, and sky exposure variations.
- In Task 2 (Resistor Bands), indoor artificial bench lighting produces severe
  cylindrical specular reflection streaks and spatial intensity falloff.

This module provides a unified radiometric and spatial preprocessing pipeline:
1. Colour Space Conversion (Unit 2):
   - Vectorised exact Gonzalez & Woods RGB <-> HSI conversion (Hue, Saturation, Intensity).
   - Fast OpenCV-compatible RGB <-> HSV and RGB <-> CIE LAB transformations.
   - Decouples chromaticity (H, S) from illumination (I / V / L).
2. Contrast Normalisation (Unit 4):
   - Pure NumPy implementation of Discrete CDF (Cumulative Distribution Function)
     Histogram Equalisation.
   - Contrast Limited Adaptive Histogram Equalisation (CLAHE) to handle local
     illumination gradients without over-amplifying sensor noise.
   - Applied exclusively to the luminance channel (I/V/L) to preserve chromatic fidelity.
3. Spatial Denoising (Unit 5):
   - Non-linear Median Filtering (3x3 or 5x5) to eliminate impulse noise and narrow
     cylindrical specular glints without blurring critical object edges.
   - 2D Gaussian Kernel smoothing to attenuate high-frequency sensor noise.
4. Comprehensive Diagnostics:
   - Publication-quality multi-panel diagnostic plotting routines comparing raw vs
     preprocessed images, channel decompositions, and intensity CDF linearisation.
"""

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ==============================================================================
# SECTION 1: UNIT 2 - COLOUR SPACE TRANSFORMATIONS
# ==============================================================================

def rgb_to_hsi(rgb_img: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image [0, 255] or [0.0, 1.0] to HSI colour space
    using the exact Gonzalez & Woods formulation (Unit 2).

    Mathematical Formulation:
      I = (R + G + B) / 3
      S = 1 - (3 / (R + G + B)) * min(R, G, B)   [for R+G+B > 0, else 0]
      theta = arccos( 0.5 * ((R - G) + (R - B)) / (sqrt((R - G)^2 + (R - B)*(G - B)) + eps) )
      H = theta if B <= G else (2*pi - theta)
      H is normalised to [0, 1], S to [0, 1], I to [0, 1].

    Parameters
    ----------
    rgb_img : np.ndarray
        Input RGB image of shape (H, W, 3), dtype uint8 or float32/float64.

    Returns
    -------
    hsi_img : np.ndarray
        Output HSI image of shape (H, W, 3), dtype float32 with values:
        - H: Hue in range [0.0, 1.0] (where 1.0 corresponds to 360 degrees)
        - S: Saturation in range [0.0, 1.0]
        - I: Intensity in range [0.0, 1.0]
    """
    if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got shape {rgb_img.shape}")

    # Normalise input to float32 in range [0.0, 1.0]
    if rgb_img.dtype == np.uint8:
        rgb = rgb_img.astype(np.float32) / 255.0
    else:
        rgb = rgb_img.astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    eps = 1e-7

    # --- 1. Intensity (I) ---
    intensity = (r + g + b) / 3.0

    # --- 2. Saturation (S) ---
    rgb_sum = r + g + b
    min_rgb = np.minimum(np.minimum(r, g), b)
    # Avoid division by zero where rgb_sum == 0
    saturation = np.zeros_like(intensity)
    non_zero_sum = rgb_sum > eps
    saturation[non_zero_sum] = 1.0 - (3.0 / rgb_sum[non_zero_sum]) * min_rgb[non_zero_sum]
    saturation = np.clip(saturation, 0.0, 1.0)

    # --- 3. Hue (H) ---
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + eps
    
    # Clip argument of arccos to [-1.0, 1.0] to prevent numerical NaNs
    arg = np.clip(num / den, -1.0, 1.0)
    theta = np.arccos(arg)  # theta in [0, pi]

    hue = np.zeros_like(intensity)
    # If B <= G, H = theta; if B > G, H = 2*pi - theta
    b_greater_g = b > g
    hue[~b_greater_g] = theta[~b_greater_g]
    hue[b_greater_g] = (2.0 * np.pi) - theta[b_greater_g]

    # Normalise Hue to [0.0, 1.0] (0 to 360 degrees)
    hue = hue / (2.0 * np.pi)

    # For achromatic pixels (saturation == 0 or rgb_sum == 0), Hue is undefined (set to 0)
    hue[saturation < eps] = 0.0

    hsi = np.stack([hue, saturation, intensity], axis=-1)
    return hsi.astype(np.float32)


def hsi_to_rgb(hsi_img: np.ndarray) -> np.ndarray:
    """
    Converts an HSI image with channels in range [0.0, 1.0] back to RGB [0, 255]
    using the exact Gonzalez & Woods sectoral reconstruction equations (Unit 2).

    Sectors:
      - Sector 1: 0 <= H < 120 deg (0 <= H < 1/3)
      - Sector 2: 120 <= H < 240 deg (1/3 <= H < 2/3)
      - Sector 3: 240 <= H < 360 deg (2/3 <= H <= 1.0)

    Parameters
    ----------
    hsi_img : np.ndarray
        Input HSI image of shape (H, W, 3), values in range [0.0, 1.0].

    Returns
    -------
    rgb_img : np.ndarray
        Reconstructed RGB image of shape (H, W, 3), dtype uint8 [0, 255].
    """
    h = hsi_img[:, :, 0] * (2.0 * np.pi)  # Convert to radians [0, 2*pi]
    s = np.clip(hsi_img[:, :, 1], 0.0, 1.0)
    i = np.clip(hsi_img[:, :, 2], 0.0, 1.0)

    r = np.zeros_like(i)
    g = np.zeros_like(i)
    b = np.zeros_like(i)

    rad_120 = 2.0 * np.pi / 3.0
    rad_240 = 4.0 * np.pi / 3.0

    # Sector 1: 0 <= H < 120 deg
    sec1 = (h >= 0.0) & (h < rad_120)
    b[sec1] = i[sec1] * (1.0 - s[sec1])
    cos_denom1 = np.cos(np.pi / 3.0 - h[sec1])
    cos_denom1 = np.where(np.abs(cos_denom1) < 1e-7, 1e-7, cos_denom1)
    r[sec1] = i[sec1] * (1.0 + (s[sec1] * np.cos(h[sec1])) / cos_denom1)
    g[sec1] = 3.0 * i[sec1] - (r[sec1] + b[sec1])

    # Sector 2: 120 <= H < 240 deg
    sec2 = (h >= rad_120) & (h < rad_240)
    h_sec2 = h[sec2] - rad_120
    r[sec2] = i[sec2] * (1.0 - s[sec2])
    cos_denom2 = np.cos(np.pi / 3.0 - h_sec2)
    cos_denom2 = np.where(np.abs(cos_denom2) < 1e-7, 1e-7, cos_denom2)
    g[sec2] = i[sec2] * (1.0 + (s[sec2] * np.cos(h_sec2)) / cos_denom2)
    b[sec2] = 3.0 * i[sec2] - (r[sec2] + g[sec2])

    # Sector 3: 240 <= H < 360 deg
    sec3 = (h >= rad_240)
    h_sec3 = h[sec3] - rad_240
    g[sec3] = i[sec3] * (1.0 - s[sec3])
    cos_denom3 = np.cos(np.pi / 3.0 - h_sec3)
    cos_denom3 = np.where(np.abs(cos_denom3) < 1e-7, 1e-7, cos_denom3)
    b[sec3] = i[sec3] * (1.0 + (s[sec3] * np.cos(h_sec3)) / cos_denom3)
    r[sec3] = 3.0 * i[sec3] - (g[sec3] + b[sec3])

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).round().astype(np.uint8)


def rgb_to_hsv_opencv(rgb_img: np.ndarray) -> np.ndarray:
    """Converts RGB uint8 image to HSV using OpenCV."""
    return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)


def hsv_to_rgb_opencv(hsv_img: np.ndarray) -> np.ndarray:
    """Converts HSV image back to RGB using OpenCV."""
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)


def rgb_to_lab_opencv(rgb_img: np.ndarray) -> np.ndarray:
    """Converts RGB uint8 image to CIE LAB using OpenCV."""
    return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)


def lab_to_rgb_opencv(lab_img: np.ndarray) -> np.ndarray:
    """Converts CIE LAB image back to RGB using OpenCV."""
    return cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)


# ==============================================================================
# SECTION 2: UNIT 4 - CONTRAST NORMALISATION & HISTOGRAM EQUALISATION
# ==============================================================================

def histogram_equalization_cdf(channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies Global Histogram Equalisation to a single-channel 8-bit image
    from first principles using the Cumulative Distribution Function (CDF) mapping (Unit 4).

    Mathematical Formulation:
      p(r_k) = n_k / (M * N)
      CDF(k) = sum_{j=0}^k p(r_j)
      s_k = round( (CDF(k) - CDF_min) / (1 - CDF_min) * (L - 1) )
      where L = 256 for 8-bit depth.

    Parameters
    ----------
    channel : np.ndarray
        Single-channel 2D array, dtype uint8 [0, 255].

    Returns
    -------
    equalized_channel : np.ndarray
        Equalised 2D array, dtype uint8.
    hist : np.ndarray
        Original histogram of shape (256,).
    cdf_normalized : np.ndarray
        Normalised CDF curve of shape (256,), values in [0, 1].
    """
    if channel.ndim != 2:
        raise ValueError(f"Expected 2D single channel, got shape {channel.shape}")
    if channel.dtype != np.uint8:
        channel = np.clip(channel, 0, 255).astype(np.uint8)

    # Compute discrete histogram
    hist, _ = np.histogram(channel.flatten(), bins=256, range=[0, 256])

    # Cumulative Distribution Function (CDF)
    cdf = hist.cumsum()
    cdf_max = cdf[-1]
    cdf_normalized = cdf / float(cdf_max)

    # Masked CDF mapping to eliminate flat minimum offset
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_min = cdf_m.min()
    
    if cdf_max > cdf_min:
        # Standard dynamic range stretching mapping
        lut = (cdf_m - cdf_min) * 255.0 / (cdf_max - cdf_min)
    else:
        lut = cdf_m * 0

    lut = np.ma.filled(lut, 0).round().astype(np.uint8)

    # Apply Look-Up Table (LUT) transformation
    equalized_channel = lut[channel]
    return equalized_channel, hist, cdf_normalized


def apply_clahe(channel: np.ndarray,
                clip_limit: float = 2.5,
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalisation (CLAHE) to a
    single-channel 8-bit image (Unit 4).

    Parameters
    ----------
    channel : np.ndarray
        Single-channel 2D array, dtype uint8.
    clip_limit : float
        Threshold for contrast limiting. Prevents noise amplification in flat regions.
    tile_grid_size : Tuple[int, int]
        Size of the contextual grid tiles (default: 8x8).

    Returns
    -------
    clahe_channel : np.ndarray
        Locally contrast-normalised 2D array, dtype uint8.
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

    Engineering Rationale:
      - Median Filter (Unit 5): Non-linear rank-order filter. It completely eliminates
        impulse noise (salt-and-pepper) and narrow specular reflection glints without
        blurring sharp colour boundaries (unlike a linear averaging filter, which causes
        edge bleeding).
      - Gaussian Filter (Unit 5): Smooths high-frequency sensor noise using a symmetric
        2D bell-shaped kernel.

    Parameters
    ----------
    channel : np.ndarray
        Input 2D channel or 3D image, dtype uint8.
    median_ksize : int
        Median filter kernel size (must be an odd integer, e.g. 3 or 5).
    gaussian_ksize : int
        Gaussian filter kernel size (must be an odd integer, e.g. 3 or 5).
    gaussian_sigma : float
        Gaussian standard deviation (sigma).

    Returns
    -------
    denoised_channel : np.ndarray
        Spatially filtered channel or image.
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
# SECTION 4: UNIFIED CONFIGURATIONS & PREPROCESSING PIPELINE
# ==============================================================================

# Default configuration dictionaries tailored to each domain's physical challenges:
DEFAULT_APPLE_CONFIG: Dict[str, Any] = {
    "colour_space": "hsi",          # 'hsi', 'hsv', or 'lab'
    "equalisation_method": "clahe",  # 'clahe', 'global_he', or 'none'
    "clahe_clip_limit": 2.5,        # Prevents blowing out sunlit foliage
    "clahe_tile_grid_size": (8, 8), # Local adaptive tiles
    "median_ksize": 3,              # 3x3 median: preserves tiny distant apple edges while removing leaf speckle
    "gaussian_ksize": 3,            # 3x3 Gaussian smoothing
    "gaussian_sigma": 0.8,
    "denoise_target": "intensity",  # Denoise intensity channel to preserve chroma transitions
}

DEFAULT_RESISTOR_CONFIG: Dict[str, Any] = {
    "colour_space": "hsi",          # 'hsi', 'hsv', or 'lab'
    "equalisation_method": "clahe",  # 'clahe', 'global_he', or 'none'
    "clahe_clip_limit": 2.0,        # Controlled contrast on resistor bodies
    "clahe_tile_grid_size": (8, 8),
    "median_ksize": 5,              # 5x5 median: critical for removing specular glare streaks along cylinder
    "gaussian_ksize": 3,            # 3x3 Gaussian smoothing
    "gaussian_sigma": 1.0,
    "denoise_target": "intensity",  # Denoise intensity channel
}


class Stage1Preprocessor:
    """
    Unified Stage 1 Preprocessing Class.
    Accepts task-specific configurations for both Orchard Apple Counting
    and Multi-Resistor Colour Band Classification.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, mode: str = "apple"):
        """
        Initialises preprocessor with task-specific configuration.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary. If None, uses mode default.
        mode : str
            Either 'apple' or 'resistor'.
        """
        if config is None:
            if mode == "apple":
                self.config = DEFAULT_APPLE_CONFIG.copy()
            elif mode == "resistor":
                self.config = DEFAULT_RESISTOR_CONFIG.copy()
            else:
                self.config = DEFAULT_APPLE_CONFIG.copy()
        else:
            self.config = config.copy()

    def process(self, rgb_img: np.ndarray) -> Dict[str, Any]:
        """
        Executes Stage 1 preprocessing on an input RGB image.

        Pipeline Stages:
          1. Validation & Colour Space Decomposition (RGB -> HSI / HSV / LAB).
          2. Contrast Normalisation (CLAHE / Global HE) on the Luminance channel.
          3. Spatial Denoising (Median + Gaussian) on the Luminance channel.
          4. Re-synthesis of Preprocessed RGB image for downstream visualisation.

        Parameters
        ----------
        rgb_img : np.ndarray
            Input RGB image, shape (H, W, 3), dtype uint8.

        Returns
        -------
        result : dict
            Dictionary containing:
            - 'original_rgb': Original input image.
            - 'preprocessed_rgb': Radiometrically normalised and denoised RGB image.
            - 'colour_space': Name of colour space used.
            - 'channel_1': 2D array (e.g. Hue [0, 255] or float).
            - 'channel_2': 2D array (e.g. Saturation [0, 255] or float).
            - 'channel_3_raw': Raw Intensity / Value / Luminance [0, 255].
            - 'channel_3_eq': Contrast-equalised Intensity / Value / Luminance [0, 255].
            - 'channel_3_denoised': Final denoised Intensity / Value / Luminance [0, 255].
            - 'raw_hist': 256-bin histogram of raw luminance.
            - 'raw_cdf': Normalised CDF of raw luminance.
            - 'eq_hist': 256-bin histogram of equalised luminance.
            - 'eq_cdf': Normalised CDF of equalised luminance.
        """
        if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB image, got shape {rgb_img.shape}")

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
        denoise_target = self.config.get("denoise_target", "intensity")

        # Step 1: Colour Space Decomposition
        if cspace == "hsi":
            hsi = rgb_to_hsi(rgb_uint8)
            h = hsi[:, :, 0]  # [0.0, 1.0]
            s = hsi[:, :, 1]  # [0.0, 1.0]
            i_float = hsi[:, :, 2]  # [0.0, 1.0]
            ch3_raw_u8 = (i_float * 255.0).round().astype(np.uint8)
            ch1_display = (h * 255.0).round().astype(np.uint8)
            ch2_display = (s * 255.0).round().astype(np.uint8)
        elif cspace == "hsv":
            hsv = rgb_to_hsv_opencv(rgb_uint8)
            ch1_display = hsv[:, :, 0]
            ch2_display = hsv[:, :, 1]
            ch3_raw_u8 = hsv[:, :, 2]
        elif cspace == "lab":
            lab = rgb_to_lab_opencv(rgb_uint8)
            ch3_raw_u8 = lab[:, :, 0]  # L channel
            ch1_display = lab[:, :, 1]  # A channel
            ch2_display = lab[:, :, 2]  # B channel
        else:
            raise ValueError(f"Unknown colour space: {cspace}")

        # Compute raw histogram and CDF
        raw_eq, raw_hist, raw_cdf = histogram_equalization_cdf(ch3_raw_u8)

        # Step 2: Contrast Normalisation on Luminance / Intensity
        if eq_method == "clahe":
            ch3_eq = apply_clahe(ch3_raw_u8, clip_limit=clip_limit, tile_grid_size=tile_grid)
        elif eq_method == "global_he":
            ch3_eq = raw_eq
        elif eq_method == "none":
            ch3_eq = ch3_raw_u8.copy()
        else:
            raise ValueError(f"Unknown equalisation method: {eq_method}")

        # Step 3: Spatial Denoising
        if denoise_target == "intensity":
            ch3_denoised = apply_spatial_denoising(
                ch3_eq,
                median_ksize=med_k,
                gaussian_ksize=gauss_k,
                gaussian_sigma=gauss_s
            )
        else:
            ch3_denoised = ch3_eq.copy()

        # Compute equalised/denoised histogram and CDF
        eq_final, eq_hist, eq_cdf = histogram_equalization_cdf(ch3_denoised)

        # Step 4: Re-synthesise RGB representation
        if cspace == "hsi":
            # Reconstruct HSI with denoised intensity
            hsi_processed = np.stack([h, s, ch3_denoised.astype(np.float32) / 255.0], axis=-1)
            preprocessed_rgb = hsi_to_rgb(hsi_processed)
            if denoise_target == "all_channels":
                preprocessed_rgb = apply_spatial_denoising(
                    preprocessed_rgb,
                    median_ksize=med_k,
                    gaussian_ksize=gauss_k,
                    gaussian_sigma=gauss_s
                )
        elif cspace == "hsv":
            hsv_processed = np.stack([ch1_display, ch2_display, ch3_denoised], axis=-1)
            preprocessed_rgb = hsv_to_rgb_opencv(hsv_processed)
            if denoise_target == "all_channels":
                preprocessed_rgb = apply_spatial_denoising(
                    preprocessed_rgb,
                    median_ksize=med_k,
                    gaussian_ksize=gauss_k,
                    gaussian_sigma=gauss_s
                )
        elif cspace == "lab":
            lab_processed = np.stack([ch3_denoised, ch1_display, ch2_display], axis=-1)
            preprocessed_rgb = lab_to_rgb_opencv(lab_processed)
            if denoise_target == "all_channels":
                preprocessed_rgb = apply_spatial_denoising(
                    preprocessed_rgb,
                    median_ksize=med_k,
                    gaussian_ksize=gauss_k,
                    gaussian_sigma=gauss_s
                )

        return {
            "original_rgb": rgb_uint8,
            "preprocessed_rgb": preprocessed_rgb,
            "colour_space": cspace,
            "channel_1": ch1_display,
            "channel_2": ch2_display,
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
    """
    Unified functional interface for Stage 1 Preprocessing.

    Parameters
    ----------
    rgb_img : np.ndarray
        Input RGB image (H, W, 3) uint8.
    mode : str
        'apple' for Orchard Apple Counting, 'resistor' for Multi-Resistor Colour Bands.
    config : dict, optional
        Custom override parameters.

    Returns
    -------
    result : dict
        Stage 1 preprocessed outputs, intermediate channels, and diagnostic histograms.
    """
    preprocessor = Stage1Preprocessor(config=config, mode=mode)
    return preprocessor.process(rgb_img)


# ==============================================================================
# SECTION 5: DIAGNOSTIC VISUALISATION ROUTINES (FOR REPORT & COLAB)
# ==============================================================================

def plot_stage1_diagnostics(result: Dict[str, Any],
                            title: str = "Stage 1 Preprocessing Diagnostics",
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Renders publication-ready multi-panel diagnostic figure showcasing
    Stage 1 transformations (ideal for Report Section 2.2 / 2.3 and Colab demo).

    Panels:
      (1) Original RGB image
      (2) Preprocessed RGB image
      (3) Hue Channel (H) / Chrominance A
      (4) Saturation Channel (S) / Chrominance B
      (5) Raw Luminance / Intensity Channel
      (6) Equalised + Denoised Intensity Channel
      (7) Histogram & CDF Before Equalisation
      (8) Histogram & CDF After Equalisation (showing CDF linearisation)

    Parameters
    ----------
    result : dict
        Output dictionary from preprocess_image() or Stage1Preprocessor.process().
    title : str
        Main title for the diagnostic figure.
    save_path : str, optional
        If specified, saves the figure to this file path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # 1. Original RGB
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(result["original_rgb"])
    ax1.set_title("1. Original RGB", fontsize=11, fontweight="semibold")
    ax1.axis("off")

    # 2. Preprocessed RGB
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(result["preprocessed_rgb"])
    ax2.set_title("2. Preprocessed RGB", fontsize=11, fontweight="semibold")
    ax2.axis("off")

    # 3. Channel 1 (Hue / A)
    ax3 = fig.add_subplot(2, 4, 3)
    im3 = ax3.imshow(result["channel_1"], cmap="hsv" if result["colour_space"] in ("hsi", "hsv") else "coolwarm")
    ax3.set_title(f"3. {result['colour_space'].upper()} Ch 1 (Hue/A)", fontsize=11, fontweight="semibold")
    ax3.axis("off")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # 4. Channel 2 (Saturation / B)
    ax4 = fig.add_subplot(2, 4, 4)
    im4 = ax4.imshow(result["channel_2"], cmap="plasma")
    ax4.set_title(f"4. {result['colour_space'].upper()} Ch 2 (Sat/B)", fontsize=11, fontweight="semibold")
    ax4.axis("off")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # 5. Raw Intensity / Luminance
    ax5 = fig.add_subplot(2, 4, 5)
    im5 = ax5.imshow(result["channel_3_raw"], cmap="gray")
    ax5.set_title("5. Raw Intensity / Luminance", fontsize=11, fontweight="semibold")
    ax5.axis("off")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    # 6. Equalised & Denoised Intensity
    ax6 = fig.add_subplot(2, 4, 6)
    im6 = ax6.imshow(result["channel_3_denoised"], cmap="gray")
    ax6.set_title("6. CLAHE + Denoised Intensity", fontsize=11, fontweight="semibold")
    ax6.axis("off")
    fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    # 7. Raw Histogram & CDF
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.bar(range(256), result["raw_hist"], color="steelblue", alpha=0.6, width=1.0, label="Intensity PDF")
    ax7.set_xlim([0, 255])
    ax7.set_xlabel("Intensity Level $r_k$", fontsize=9)
    ax7.set_ylabel("Pixel Frequency", fontsize=9, color="steelblue")
    ax7.tick_params(axis="y", labelcolor="steelblue")

    ax7_twin = ax7.twinx()
    ax7_twin.plot(range(256), result["raw_cdf"], color="darkred", linewidth=2.0, label="CDF")
    ax7_twin.set_ylim([0.0, 1.05])
    ax7_twin.set_ylabel("Normalised CDF", fontsize=9, color="darkred")
    ax7_twin.tick_params(axis="y", labelcolor="darkred")
    ax7.set_title("7. Raw Histogram & CDF", fontsize=11, fontweight="semibold")

    # 8. Equalised Histogram & CDF
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.bar(range(256), result["eq_hist"], color="forestgreen", alpha=0.6, width=1.0, label="Intensity PDF")
    ax8.set_xlim([0, 255])
    ax8.set_xlabel("Intensity Level $s_k$", fontsize=9)
    ax8.set_ylabel("Pixel Frequency", fontsize=9, color="forestgreen")
    ax8.tick_params(axis="y", labelcolor="forestgreen")

    ax8_twin = ax8.twinx()
    ax8_twin.plot(range(256), result["eq_cdf"], color="darkmagenta", linewidth=2.0, label="Linearised CDF")
    ax8_twin.set_ylim([0.0, 1.05])
    ax8_twin.set_ylabel("Normalised CDF", fontsize=9, color="darkmagenta")
    ax8_twin.tick_params(axis="y", labelcolor="darkmagenta")
    ax8.set_title("8. Equalised Histogram & CDF", fontsize=11, fontweight="semibold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_specular_line_profile_comparison(raw_channel: np.ndarray,
                                          denoised_channel: np.ndarray,
                                          row_idx: int,
                                          title: str = "Specular Reflection Glint Suppression (Unit 5)",
                                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Renders 1D cross-sectional line profile showing how median filtering suppresses
    specular reflection spikes on cylindrical resistors or apple skin glints without
    blurring the true step transitions of colour bands or fruit boundaries.

    Parameters
    ----------
    raw_channel : np.ndarray
        2D intensity array before denoising.
    denoised_channel : np.ndarray
        2D intensity array after denoising.
    row_idx : int
        Horizontal line index through the specular streak.
    title : str
        Figure title.
    save_path : str, optional
        Save path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(raw_channel.shape[1])
    ax.plot(x, raw_channel[row_idx, :], color="red", linestyle="--", alpha=0.8, label="Raw Intensity (with Glare Spikes)")
    ax.plot(x, denoised_channel[row_idx, :], color="blue", linewidth=1.8, label="Median Denoised (Specular Glint Suppressed)")
    ax.set_title(f"{title} (Row {row_idx})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Horizontal Pixel Coordinate (x)", fontsize=10)
    ax.set_ylabel("Intensity Level (0-255)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
