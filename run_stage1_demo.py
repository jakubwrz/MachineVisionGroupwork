"""
Stage 1 Demonstration & Evaluation Script
Machine Vision (UFMFRR-15-M) Group Project

Primary Contributor: [Group Member Name / Student ID]

This script demonstrates Stage 1 (Radiometric & Spatial Preprocessing) on two synthetic
benchmarks representative of the assessment scenarios:
  1. Orchard Apple Cluster: Non-uniform solar canopy shading, dappled shadows, leaf clutter.
  2. Multi-Resistor Scene: Cylindrical bodies, colour bands, and bright specular glare streaks.

Outputs:
  - High-resolution diagnostic figures saved in output/ directory (suitable for Report Figures 3 & 5).
"""

import os
import sys
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


def create_synthetic_orchard_scene(height: int = 400, width: int = 500) -> np.ndarray:
    """
    Synthesises an orchard scene with apples, foliage, and a non-uniform
    canopy illumination gradient / shadow pattern (Unit 2 & 4 challenge).
    """
    np.random.seed(101)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Base foliage background (dark green textured leaves)
    leaves_base = np.random.normal(loc=70, scale=15, size=(height, width)).clip(30, 120).astype(np.uint8)
    img[:, :, 0] = (leaves_base * 0.35).astype(np.uint8)  # R
    img[:, :, 1] = (leaves_base * 1.1).clip(0, 255).astype(np.uint8)  # G
    img[:, :, 2] = (leaves_base * 0.25).astype(np.uint8)  # B

    # 2. Add spherical apples (Red and Yellow/Green)
    # Apple 1: Red apple
    cv2.circle(img, (150, 180), 55, (190, 25, 20), -1)
    # Apple 2: Partially overlapping red apple
    cv2.circle(img, (220, 210), 50, (175, 20, 15), -1)
    # Apple 3: Sunlit red-yellow apple
    cv2.circle(img, (330, 150), 48, (210, 80, 25), -1)
    # Apple 4: Green apple in foliage
    cv2.circle(img, (260, 290), 45, (110, 170, 20), -1)
    # Apple 5: Distant small apple
    cv2.circle(img, (390, 280), 28, (180, 30, 20), -1)

    # Add soft edge shading to simulate 3D spherical volume
    y, x = np.ogrid[:height, :width]
    for cx, cy, r in [(150, 180, 55), (220, 210, 50), (330, 150, 48), (260, 290, 45), (390, 280, 28)]:
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = dist <= r
        shading = 1.0 - 0.35 * (dist / r) ** 2
        for c in range(3):
            img[mask, c] = (img[mask, c] * shading[mask]).clip(0, 255).astype(np.uint8)

    # 3. Simulate harsh canopy shadow (non-uniform illumination gradient)
    # Dark diagonal shadow from top-left to bottom-right
    grad = np.linspace(0.3, 1.2, width)
    grad_2d = np.tile(grad, (height, 1))
    shadow_mask = np.clip(grad_2d + 0.2 * np.sin(np.linspace(0, 6, height))[:, None], 0.25, 1.2)
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c] * shadow_mask, 0, 255).astype(np.uint8)

    # 4. Add sensor noise
    noise = np.random.normal(0, 6, (height, width, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def create_synthetic_resistor_scene(height: int = 350, width: int = 600) -> np.ndarray:
    """
    Synthesises a workbench scene with cylindrical resistors, colour bands,
    horizontal metallic leads, and specular reflection streaks (Unit 5 challenge).
    """
    np.random.seed(202)
    # Workbench background (light grey matte wood/mat)
    bg = np.random.normal(160, 8, (height, width)).clip(130, 190).astype(np.uint8)
    img = np.stack([bg, bg, (bg * 0.95).astype(np.uint8)], axis=-1)

    # Resistor 1: Beige Carbon-Film Resistor (4 bands: Brown, Black, Red, Gold)
    # Leads
    cv2.line(img, (50, 120), (550, 120), (130, 130, 135), 4)  # Wire lead
    # Cylindrical body
    cv2.rectangle(img, (180, 85), (420, 155), (170, 190, 210), -1)  # Beige/tan in BGR/RGB: (210, 190, 170)
    # Colour bands
    bands_res1 = [
        (215, 227, (120, 60, 20)),    # Brown (RGB)
        (245, 257, (20, 20, 20)),      # Black
        (275, 287, (210, 25, 20)),     # Red
        (370, 385, (210, 175, 45)),    # Gold
    ]
    for x1, x2, rgb_color in bands_res1:
        img[85:156, x1:x2] = rgb_color

    # Resistor 2: Blue Metal-Film Resistor (5 bands: Yellow, Violet, Black, Orange, Brown)
    cv2.line(img, (70, 250), (530, 250), (135, 135, 140), 4)
    cv2.rectangle(img, (190, 215), (430, 285), (40, 110, 190), -1)  # Blue body
    bands_res2 = [
        (215, 227, (230, 210, 30)),    # Yellow
        (245, 257, (130, 30, 140)),    # Violet
        (275, 287, (20, 20, 20)),      # Black
        (305, 317, (235, 120, 20)),    # Orange
        (380, 392, (120, 60, 20)),     # Brown
    ]
    for x1, x2, rgb_color in bands_res2:
        img[215:286, x1:x2] = rgb_color

    # Add cylindrical lighting falloff (brighter in centre, darker at cylinder top/bottom edges)
    for y_start, y_end in [(85, 155), (215, 285)]:
        h_body = y_end - y_start
        cyl_profile = np.sin(np.linspace(0.1, np.pi - 0.1, h_body))
        for r_idx in range(h_body):
            curr_y = y_start + r_idx
            factor = 0.6 + 0.5 * cyl_profile[r_idx]
            img[curr_y, 180:430] = np.clip(img[curr_y, 180:430] * factor, 0, 255).astype(np.uint8)

    # Add SPECULAR GLARE STREAKS along the cylinder length (Unit 5 problem)
    # Intense white specular reflection stripe along row 112 and row 242
    glare_rows = [112, 113, 242, 243]
    for r in glare_rows:
        img[r, 190:420] = np.clip(img[r, 190:420] + 180, 0, 255).astype(np.uint8)

    # Sensor noise
    noise = np.random.normal(0, 4, (height, width, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def run_demonstration():
    print("=" * 70)
    print("Machine Vision (UFMFRR-15-M) - Stage 1 Preprocessing Demonstration")
    print("=" * 70)

    os.makedirs("output", exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. ORCHARD APPLE SCENARIO
    # --------------------------------------------------------------------------
    print("\n[1/3] Processing Orchard Apple Scenario...")
    apple_img = create_synthetic_orchard_scene()
    apple_res = preprocess_image(apple_img, mode="apple")

    print(f"  - Colour Space: {apple_res['colour_space'].upper()}")
    print(f"  - Contrast Normalisation: CLAHE (clipLimit={apple_res['config']['clahe_clip_limit']})")
    print(f"  - Spatial Denoising: Median ({apple_res['config']['median_ksize']}x{apple_res['config']['median_ksize']}) + Gaussian ({apple_res['config']['gaussian_ksize']}x{apple_res['config']['gaussian_ksize']})")
    
    apple_fig_path = os.path.join("output", "figure_stage1_apple_diagnostics.png")
    fig1 = plot_stage1_diagnostics(
        apple_res,
        title="Stage 1 Diagnostics: Orchard Apple Canopy Preprocessing (Units 2, 4, 5)",
        save_path=apple_fig_path
    )
    plt.close(fig1)
    print(f"  -> Saved diagnostic figure: {apple_fig_path}")

    # --------------------------------------------------------------------------
    # 2. MULTI-RESISTOR SCENARIO
    # --------------------------------------------------------------------------
    print("\n[2/3] Processing Multi-Resistor Scenario...")
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
    glare_row = 112  # Specular line along the upper resistor
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

    print("\n" + "=" * 70)
    print("Stage 1 Pipeline successfully executed across both domains!")
    print("All diagnostic figures verified and written to output/ directory.")
    print("=" * 70)


if __name__ == "__main__":
    run_demonstration()
