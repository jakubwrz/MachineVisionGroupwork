"""
Machine Vision (UFMFRR-15-M) - Group Project
Stage 1: Radiometric & Spatial Preprocessing (Units 2, 4, 5)

Primary Contributor: [Name]
"""

import os
import glob
import cv2
import matplotlib.pyplot as plt


def preprocess_stage1(rgb_img, mode="apple"):
    """
    Stage 1 Unified Preprocessing:
      1. Colour Space Conversion (Unit 2): Decouples chrominance from luminance via HSV/HSI.
      2. Contrast Normalisation (Unit 4): Applies CLAHE exclusively to the Intensity channel.
      3. Spatial Denoising (Unit 5): Applies Median filtering (3x3 for apples, 5x5 for resistors)
         followed by 3x3 Gaussian smoothing.
    """
    # 1. Colour Space Conversion (Unit 2)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # 2. Contrast Normalisation on Intensity ONLY (Unit 4)
    clip = 2.5 if mode == "apple" else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)

    # 3. Spatial Denoising (Unit 5)
    k_med = 3 if mode == "apple" else 5
    v_denoised = cv2.medianBlur(v_eq, k_med)
    v_denoised = cv2.GaussianBlur(v_denoised, (3, 3), 0.8)

    # Recombine with untouched Hue and Saturation
    preprocessed_rgb = cv2.cvtColor(cv2.merge([h, s, v_denoised]), cv2.COLOR_HSV2RGB)
    return preprocessed_rgb, h, s, v_denoised


def run_stage1():
    print("=" * 60)
    print("Machine Vision (UFMFRR-15-M) - Stage 1 Preprocessing")
    print("=" * 60)

    # Locate dataset
    img_files = (
        glob.glob("minneapple-1-1/train/*.jpg") or
        glob.glob("/content/minneapple-1-1/train/*.jpg") or
        glob.glob("**/train/*.jpg", recursive=True)
    )

    if not img_files:
        print("[ERROR] Could not find dataset images in minneapple-1-1/train/.")
        return

    # Load and process sample image
    img_path = img_files[0]
    raw_bgr = cv2.imread(img_path)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    print(f"Loaded: {os.path.basename(img_path)} (Shape: {raw_rgb.shape})")

    clean_rgb, h, s, v = preprocess_stage1(raw_rgb, mode="apple")

    # Side-by-side Before vs After plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(raw_rgb)
    plt.title(f"Original RGB: {os.path.basename(img_path)}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(clean_rgb)
    plt.title("Stage 1 Preprocessed (CLAHE + Median Denoised)")
    plt.axis("off")

    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "stage1_output.png")
    plt.savefig(out_path, bbox_inches="tight")

    # Display plot in Colab/Jupyter if interactive
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
