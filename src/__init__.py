"""
Machine Vision (UFMFRR-15-M) Group Project
Source Code Package
"""

from .stage1_preprocessing import (
    rgb_to_hsi,
    hsi_to_rgb,
    rgb_to_hsv_opencv,
    hsv_to_rgb_opencv,
    rgb_to_lab_opencv,
    lab_to_rgb_opencv,
    histogram_equalization_cdf,
    apply_clahe,
    apply_spatial_denoising,
    Stage1Preprocessor,
    preprocess_image,
    plot_stage1_diagnostics,
    plot_specular_line_profile_comparison,
    DEFAULT_APPLE_CONFIG,
    DEFAULT_RESISTOR_CONFIG,
)

from .minneapple_dataset import (
    MinneAppleDataset,
    process_minneapple_batch,
)

__all__ = [
    "rgb_to_hsi",
    "hsi_to_rgb",
    "rgb_to_hsv_opencv",
    "hsv_to_rgb_opencv",
    "rgb_to_lab_opencv",
    "lab_to_rgb_opencv",
    "histogram_equalization_cdf",
    "apply_clahe",
    "apply_spatial_denoising",
    "Stage1Preprocessor",
    "preprocess_image",
    "plot_stage1_diagnostics",
    "plot_specular_line_profile_comparison",
    "DEFAULT_APPLE_CONFIG",
    "DEFAULT_RESISTOR_CONFIG",
    "MinneAppleDataset",
    "process_minneapple_batch",
]
