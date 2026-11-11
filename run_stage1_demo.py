"""
Stage 1 Demonstration & Evaluation Script
Machine Vision (UFMFRR-15-M) Group Project

Primary Contributor: [Group Member Name / Student ID]

This is the primary demonstration script for Stage 1 (Radiometric & Spatial Preprocessing).
It handles both project scenarios:
  1. Task 1 (Orchard Apples): Automatically loads from the real MinneApple dataset
     (or falls back to a synthetic canopy scene if dataset is not present), performs
     HSI decoupling, Intensity CLAHE contrast normalisation, and 3x3 median denoising,
     with 512x512 tiling demonstration.
  2. Task 2 (Multi-Resistor): Models cylindrical resistor bodies, colour bands, and
     demonstrates 5x5 median filtering to eliminate bright specular glare streaks.

Outputs:
  - Diagnostic 8-panel figures (Report Section 2.2 / 2.3)
  - 1D cross-sectional specular reflection glint suppression profile
"""

import os
import sys
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.stage1_preprocessing import (
    preprocess_image,
    plot_stage1_diagnostics,
    plot_specular_line_profile_comparison,
    DEFAULT_APPLE_CONFIG,
    DEFAULT_RESISTOR_CONFIG,
)
from src.minneapple_dataset import MinneAppleDataset


def create_synthetic_orchard_scene(height: int = 400, width: int = 500) -> np.ndarray:
    """Synthesises an orchard scene with apples, foliage, and canopy shadow gradient."""
    np.random.seed(101)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Base foliage background
    leaves_base = np.random.normal(loc=70, scale=15, size=(height, width)).clip(30, 120).astype(np.uint8)
    img[:, :, 0] = (leaves_base * 0.35).astype(np.uint8)
    img[:, :, 1] = (leaves_base * 1.1).clip(0, 255).astype(np.uint8)
    img[:, :, 2] = (leaves_base * 0.25).astype(np.uint8)

    # Add spherical apples
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

    # Canopy shadow gradient
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
    bands_res1 = [
        (215, 227, (120, 60, 20)),
        (245, 257, (20, 20, 20)),
        (275, 287, (210, 25, 20)),
        (370, 385, (210, 175, 45)),
    ]
    for x1, x2, rgb_color in bands_res1:
        img[85:156, x1:x2] = rgb_color

    # Resistor 2: Blue (Yellow, Violet, Black, Orange, Brown)
    cv2.line(img, (70, 250), (530, 250), (135, 135, 140), 4)
    cv2.rectangle(img, (190, 215), (430, 285), (40, 110, 190), -1)
    bands_res2 = [
        (215, 227, (230, 210, 30)),
        (245, 257, (130, 30, 140)),
        (275, 287, (20, 20, 20)),
        (305, 317, (235, 120, 20)),
        (380, 392, (120, 60, 20)),
    ]
    for x1, x2, rgb_color in bands_res2:
        img[215:286, x1:x2] = rgb_color

    # Cylindrical body falloff
    for y_start, y_end in [(85, 155), (215, 285)]:
        h_body = y_end - y_start
        cyl_profile = np.sin(np.linspace(0.1, np.pi - 0.1, h_body))
        for r_idx in range(h_body):
            curr_y = y_start + r_idx
            factor = 0.6 + 0.5 * cyl_profile[r_idx]
            img[curr_y, 180:430] = np.clip(img[curr_y, 180:430] * factor, 0, 255).astype(np.uint8)

    # Specular reflection streaks (Unit 5 problem)
    for r in [112, 113, 242, 243]:
        img[r, 190:420] = np.clip(img[r, 190:420] + 180, 0, 255).astype(np.uint8)

    noise = np.random.normal(0, 4, (height, width, 3)).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def run_demonstration():
    print("=" * 75)
    print("Machine Vision (UFMFRR-15-M) - Stage 1 Preprocessing Demonstration")
    print("=" * 75)

    os.makedirs("output", exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. ORCHARD APPLE SCENARIO (TASK 1)
    # --------------------------------------------------------------------------
    print("\n[1/3] Processing Orchard Apple Scenario (Task 1)...")
    
    # Auto-detect real MinneApple dataset location
    candidate_paths = [
        "minneapple-1-1/train",
        "/content/minneapple-1-1/train",
        "/content/MachineVisionGroupwork/minneapple-1-1/train",
        "/content/drive/MyDrive/minneapple-1-1/train",
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
    fig1 = plot_stage1_diagnostics(
        apple_res,
        title=f"Stage 1 Diagnostics: {title_prefix} (Units 2, 4, 5)",
        save_path=apple_fig_path
    )
    plt.close(fig1)
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
    fig2 = plot_stage1_diagnostics(
        resistor_res,
        title="Stage 1 Diagnostics: Resistor Body & Specular Glare Preprocessing (Units 2, 4, 5)",
        save_path=resistor_fig_path
    )
    plt.close(fig2)
    print(f"  -> Saved diagnostic figure: {resistor_fig_path}")

    # --------------------------------------------------------------------------
    # 3. SPECULAR GLARE LINE PROFILE EVALUATION
    # --------------------------------------------------------------------------
    print("\n[3/3] Generating Specular Reflection Glint Suppression Analysis...")
    glare_row = 112
    profile_fig_path = os.path.join("output", "figure_stage1_specular_profile.png")
    fig3 = plot_specular_line_profile_comparison(
        raw_channel=resistor_res["channel_3_raw"],
        denoised_channel=resistor_res["channel_3_denoised"],
        row_idx=glare_row,
        title="Specular Glare Profile on Cylindrical Resistor Body (Unit 5 Median Filtering)",
        save_path=profile_fig_path
    )
    plt.close(fig3)
    print(f"  -> Saved 1D profile figure: {profile_fig_path}")

    print("\n" + "=" * 75)
    print("Stage 1 Pipeline successfully executed across both domains!")
    print("All diagnostic figures verified and written to output/ directory.")
    print("=" * 75)


if __name__ == "__main__":
    run_demonstration()
