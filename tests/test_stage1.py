"""
Unit Tests for Stage 1: Radiometric & Spatial Preprocessing (Units 2, 4, 5)
Machine Vision (UFMFRR-15-M) Group Project

Runs with standard Python unittest (no external testing libraries required).
"""

import os
import sys
import unittest
import numpy as np

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage1_preprocessing import (
    rgb_to_hsi,
    hsi_to_rgb,
    histogram_equalization_cdf,
    apply_clahe,
    apply_spatial_denoising,
    Stage1Preprocessor,
    preprocess_image,
    DEFAULT_APPLE_CONFIG,
    DEFAULT_RESISTOR_CONFIG,
)


class TestStage1Preprocessing(unittest.TestCase):

    def test_rgb_to_hsi_basic_colours(self):
        """Verifies HSI conversion for fundamental pure colours (Unit 2)."""
        red = np.array([[[255, 0, 0]]], dtype=np.uint8)
        green = np.array([[[0, 255, 0]]], dtype=np.uint8)
        blue = np.array([[[0, 0, 255]]], dtype=np.uint8)
        white = np.array([[[255, 255, 255]]], dtype=np.uint8)
        black = np.array([[[0, 0, 0]]], dtype=np.uint8)

        # Red: Hue should be 0 deg (0.0)
        hsi_red = rgb_to_hsi(red)
        self.assertTrue(np.isclose(hsi_red[0, 0, 0], 0.0, atol=1e-3) or np.isclose(hsi_red[0, 0, 0], 1.0, atol=1e-3))
        self.assertTrue(np.isclose(hsi_red[0, 0, 1], 1.0, atol=1e-3))
        self.assertTrue(np.isclose(hsi_red[0, 0, 2], 1.0 / 3.0, atol=1e-3))

        # Green: Hue should be 120 deg (1/3 approx 0.3333)
        hsi_green = rgb_to_hsi(green)
        self.assertTrue(np.isclose(hsi_green[0, 0, 0], 1.0 / 3.0, atol=1e-3))
        self.assertTrue(np.isclose(hsi_green[0, 0, 1], 1.0, atol=1e-3))

        # Blue: Hue should be 240 deg (2/3 approx 0.6667)
        hsi_blue = rgb_to_hsi(blue)
        self.assertTrue(np.isclose(hsi_blue[0, 0, 0], 2.0 / 3.0, atol=1e-3))
        self.assertTrue(np.isclose(hsi_blue[0, 0, 1], 1.0, atol=1e-3))

        # White: Saturation should be 0, Intensity 1.0
        hsi_white = rgb_to_hsi(white)
        self.assertTrue(np.isclose(hsi_white[0, 0, 1], 0.0, atol=1e-3))
        self.assertTrue(np.isclose(hsi_white[0, 0, 2], 1.0, atol=1e-3))

        # Black: Saturation 0, Intensity 0.0
        hsi_black = rgb_to_hsi(black)
        self.assertTrue(np.isclose(hsi_black[0, 0, 2], 0.0, atol=1e-3))

    def test_hsi_roundtrip_reconstruction(self):
        """Verifies that RGB -> HSI -> RGB preserves pixel values accurately."""
        np.random.seed(42)
        test_img = np.random.randint(20, 240, size=(64, 64, 3), dtype=np.uint8)
        hsi = rgb_to_hsi(test_img)
        rec_rgb = hsi_to_rgb(hsi)

        diff = np.abs(test_img.astype(np.int32) - rec_rgb.astype(np.int32))
        self.assertLessEqual(np.max(diff), 3)
        self.assertLess(np.mean(diff), 0.8)

    def test_histogram_equalization_cdf(self):
        """Verifies discrete CDF histogram equalisation from first principles (Unit 4)."""
        low_contrast = np.random.randint(100, 120, size=(100, 100), dtype=np.uint8)
        eq_channel, hist, cdf = histogram_equalization_cdf(low_contrast)

        self.assertEqual(eq_channel.min(), 0)
        self.assertEqual(eq_channel.max(), 255)
        self.assertEqual(len(hist), 256)
        self.assertTrue(np.all(np.diff(cdf) >= 0))  # Monotonically non-decreasing
        self.assertTrue(np.isclose(cdf[-1], 1.0, atol=1e-5))

    def test_clahe_contrast_enhancement(self):
        """Verifies CLAHE improves local contrast and expands standard deviation."""
        np.random.seed(42)
        # Create a low-contrast image (narrow standard deviation)
        low_contrast = np.random.normal(loc=128, scale=10, size=(128, 128)).clip(0, 255).astype(np.uint8)
        clahe_out = apply_clahe(low_contrast, clip_limit=3.0, tile_grid_size=(8, 8))

        self.assertEqual(clahe_out.shape, low_contrast.shape)
        self.assertEqual(clahe_out.dtype, np.uint8)
        # Dynamic range (std dev) must be significantly expanded by CLAHE
        self.assertGreater(float(clahe_out.std()), float(low_contrast.std()) * 1.5)

    def test_median_filter_specular_and_impulse_noise(self):
        """Verifies 3x3 and 5x5 median filter removes specular spikes and salt-and-pepper noise."""
        base_channel = np.full((50, 50), 100, dtype=np.uint8)
        noisy = base_channel.copy()
        noisy[25, 10:40] = 255  # 1-pixel tall specular line

        denoised_3x3 = apply_spatial_denoising(noisy, median_ksize=3, gaussian_ksize=1)
        self.assertTrue(np.all(denoised_3x3[25, 12:38] < 200))

        noisy_glare = base_channel.copy()
        noisy_glare[24:26, 10:40] = 255
        denoised_5x5 = apply_spatial_denoising(noisy_glare, median_ksize=5, gaussian_ksize=1)
        self.assertTrue(np.all(denoised_5x5[24:26, 12:38] < 200))

    def test_unified_pipeline_modes(self):
        """Tests the unified preprocess_image interface for both apple and resistor modes."""
        np.random.seed(42)
        sample_img = np.random.randint(30, 220, size=(128, 128, 3), dtype=np.uint8)

        # Test Apple Mode
        res_apple = preprocess_image(sample_img, mode="apple")
        self.assertEqual(res_apple["colour_space"], "hsi")
        self.assertEqual(res_apple["preprocessed_rgb"].shape, sample_img.shape)
        self.assertEqual(res_apple["channel_3_denoised"].shape, (128, 128))
        self.assertEqual(res_apple["config"]["median_ksize"], 3)

        # Test Resistor Mode
        res_resistor = preprocess_image(sample_img, mode="resistor")
        self.assertEqual(res_resistor["colour_space"], "hsi")
        self.assertEqual(res_resistor["preprocessed_rgb"].shape, sample_img.shape)
        self.assertEqual(res_resistor["config"]["median_ksize"], 5)


if __name__ == "__main__":
    unittest.main()
