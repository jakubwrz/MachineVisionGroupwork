"""
Machine Vision (UFMFRR-15-M) - Group Project
Stage 1: Radiometric & Spatial Preprocessing (Units 2, 4, 5)

Primary Contributor: [Name]
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
    Stage 1 Unified Preprocessing:
      1. Luminance Decoupling & Contrast Normalisation (Units 2 & 4):
         Applies CLAHE to perceptual Luminance (LAB L-channel) to boost shadow
         detail and flatten lighting gradients without shifting Hue.
      2. Spatial Denoising (Unit 5):
         Applies Median filtering (3x3 for apples, 5x5 for resistors) and 3x3 Gaussian
         smoothing in RGB space so H and S are denoised without circular 0/179 Hue artefacts.
      3. Channel Extraction (Unit 2):
         Returns denoised RGB, clean H, S, V, and true Unit 2 Intensity I = (R+G+B)/3.
    """
    # 1. Contrast Normalisation on Luminance ONLY (Unit 2 & Unit 4 Slide 9)
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l_raw, a, b = cv2.split(lab)

    clip = 2.5 if mode == "apple" else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_raw)

    eq_rgb = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)

    # 2. Spatial Denoising across RGB (Unit 5 Slides 11-28)
    k_med = 3 if mode == "apple" else 5
    denoised_rgb = cv2.medianBlur(eq_rgb, k_med)
    preprocessed_rgb = cv2.GaussianBlur(denoised_rgb, (3, 3), 0.8)

    # 3. Extract clean HSV + true Unit 2 Intensity channels for Stage 2
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


def run_stage1():
    print("=" * 60)
    print("Machine Vision (UFMFRR-15-M) - Stage 1 Preprocessing")
    print("=" * 60)

    img_files = (
        glob.glob("minneapple-1-1/train/*.jpg") or
        glob.glob("/content/minneapple-1-1/train/*.jpg") or
        glob.glob("**/train/*.jpg", recursive=True)
    )

    if not img_files:
        print("[ERROR] Could not find dataset images in minneapple-1-1/train/.")
        return

    img_path = img_files[0]
    raw_bgr = cv2.imread(img_path)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    print(f"Loaded: {os.path.basename(img_path)} (Shape: {raw_rgb.shape})")

    clean_rgb, h, s, v, i_unit2, l_raw, l_eq = preprocess_stage1(raw_rgb, mode="apple")

    # 4-Panel Report-Ready Diagnostic Figure (Images + Unit 4 Histograms/CDFs)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].imshow(raw_rgb)
    axes[0, 0].set_title(f"(a) Raw RGB: {os.path.basename(img_path)}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(clean_rgb)
    axes[0, 1].set_title("(b) Stage 1 Output (Luminance CLAHE + RGB Median/Gaussian)")
    axes[0, 1].axis("off")

    plot_hist_and_cdf(axes[1, 0], l_raw, "(c) Raw Luminance Histogram & CDF")
    plot_hist_and_cdf(axes[1, 1], l_eq, "(d) Post-CLAHE Luminance Histogram & CDF")

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "stage1_output.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")

    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            plt.show()
        else:
            plt.close()
    except Exception:
        plt.close()

    print(f"Stage 1 successfully completed! Output saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_stage1()