"""
Machine Vision (UFMFRR-15-M) - Group Project
Stage 1: Radiometric & Spatial Preprocessing (Units 2, 4, 5)

Primary Contributor: Jakub Wrzyszcz

Unified Pipeline Methodology:
  1. Luminance Decoupling & Contrast Normalisation (Units 2 & 4):
     Applies Contrast Limited Adaptive Histogram Equalisation (CLAHE) exclusively
     to perceptual Luminance (CIE LAB L-channel) to enhance shadow details and flatten
     lighting gradients without distorting colour chromaticity (Hue / Saturation).
     - Apples (clipLimit=2.5, tileGridSize=(8, 8)): Compensates for harsh orchard canopy shadows.
     - Resistors (clipLimit=2.0, tileGridSize=(8, 8)): Flattens uneven workbench illumination.
  2. Spatial Denoising (Unit 5):
     Applies Median filtering and 3x3 Gaussian smoothing across RGB space. Denoising in RGB
     avoids phase wrap-around artefacts at the 0/179 Hue boundary.
     - Apples (3x3 median): Suppresses fine sensor noise while preserving edge boundaries.
     - Resistors (5x5 median): Attenuates specular glare streaks along cylindrical bodies.
  3. Channel Representation (Unit 2):
     Extracts denoised RGB, clean HSV channels, and true Unit 2 Intensity:
     I = (R + G + B) / 3.
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_unit2_intensity(rgb_img):
    """Computes true Unit 2 Slide 25 Intensity: I = (R + G + B) / 3."""
    return np.mean(rgb_img.astype(np.float32), axis=2).astype(np.uint8)


def preprocess_stage1(rgb_img, mode="apple"):
    """
    Stage 1 Unified Preprocessing for both Apples and Resistors:
      - Normalizes contrast via CLAHE on perceptual Luminance (LAB L).
      - Denoises spatially using median + Gaussian smoothing.
      - Returns preprocessed RGB, H, S, V, true Unit 2 Intensity I, raw L, and equalised L.
    """
    # 1. Decoupled Contrast Normalisation on Luminance (Units 2 & 4)
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l_raw, a_chan, b_chan = cv2.split(lab)

    clip = 2.5 if mode == "apple" else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_raw)

    eq_rgb = cv2.cvtColor(cv2.merge([l_eq, a_chan, b_chan]), cv2.COLOR_LAB2RGB)

    # 2. Spatial Denoising in RGB (Unit 5)
    k_med = 3 if mode == "apple" else 5
    denoised_rgb = cv2.medianBlur(eq_rgb, k_med)
    preprocessed_rgb = cv2.GaussianBlur(denoised_rgb, (3, 3), 0.8)

    # 3. Channel Extraction
    hsv = cv2.cvtColor(preprocessed_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    i_unit2 = compute_unit2_intensity(preprocessed_rgb)

    return preprocessed_rgb, h, s, v, i_unit2, l_raw, l_eq


def plot_hist_and_cdf(ax, channel, title):
    """Plots normalized histogram and Cumulative Distribution Function (Unit 4 Slide 9)."""
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel()
    hist_norm = hist / hist.sum()
    cdf = hist_norm.cumsum()

    ax.bar(range(256), hist_norm, color="steelblue", alpha=0.6, width=1.0, label="PDF")
    ax_cdf = ax.twinx()
    ax_cdf.plot(cdf, color="crimson", linewidth=1.8, label="CDF")
    ax_cdf.set_ylim(0, 1.05)
    ax.set_xlim(0, 255)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Pixel Intensity [0-255]")


def process_single_stage1(img_path, mode, out_filename):
    """Loads, preprocesses, and plots 4-panel diagnostic for one dataset image."""
    raw_bgr = cv2.imread(img_path)
    if raw_bgr is None:
        print(f"[ERROR] Could not read image at {img_path}")
        return None

    # Rescale high-res images to max dimension 1280 to standardise spatial scale
    h, w = raw_bgr.shape[:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        raw_bgr = cv2.resize(raw_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    print(f"[{mode.upper()}] Loaded: {os.path.basename(img_path)} (Shape: {raw_rgb.shape})")

    clean_rgb, h_ch, s_ch, v_ch, i_ch, l_raw, l_eq = preprocess_stage1(raw_rgb, mode=mode)

    # 4-Panel Report Diagnostic Figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].imshow(raw_rgb)
    axes[0, 0].set_title(f"(a) Raw {mode.capitalize()} RGB: {os.path.basename(img_path)}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(clean_rgb)
    k_desc = "3x3" if mode == "apple" else "5x5"
    axes[0, 1].set_title(f"(b) Stage 1 Output (Luminance CLAHE + RGB {k_desc} Median/Gaussian)")
    axes[0, 1].axis("off")

    plot_hist_and_cdf(axes[1, 0], l_raw, "(c) Raw Luminance Histogram & CDF")
    plot_hist_and_cdf(axes[1, 1], l_eq, "(d) Post-CLAHE Luminance Histogram & CDF")

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", out_filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")

    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            plt.show()
        else:
            plt.close()
    except Exception:
        plt.close()

    print(f"  -> Saved Stage 1 diagnostic: {out_path}")
    return clean_rgb


def run_stage1():
    print("=" * 60)
    print("Machine Vision (UFMFRR-15-M) - Stage 1 Preprocessing (Dual Pipeline)")
    print("=" * 60)

    # 1. Apples (MinneApple Dataset)
    apple_files = (
        glob.glob("minneapple-1-1/train/*.jpg")
        or glob.glob("/content/minneapple-1-1/train/*.jpg")
        or glob.glob("**/train/*.jpg", recursive=True)
    )
    if apple_files:
        process_single_stage1(apple_files[0], mode="apple", out_filename="stage1_apple_output.png")
    else:
        print("[WARNING] No apple images found in minneapple-1-1/train/.")

    # 2. Resistors (Captured Resistor Dataset)
    resistor_files = (
        glob.glob("Resistor Dataset/*.jpg")
        or glob.glob("Resistor Dataset/*.png")
        or glob.glob("/content/Resistor Dataset/*.jpg")
        or glob.glob("**/Resistor Dataset/*.jpg", recursive=True)
    )
    if resistor_files:
        process_single_stage1(resistor_files[0], mode="resistor", out_filename="stage1_resistor_output.png")
    else:
        print("[WARNING] No resistor images found in 'Resistor Dataset'.")

    print("\nStage 1 successfully executed across both tasks!")
    print("=" * 60)


if __name__ == "__main__":
    run_stage1()
