"""
Stage 1 Execution Script for the MinneApple Dataset
Machine Vision (UFMFRR-15-M) Group Project

Primary Contributor: [Group Member Name / Student ID]
Task 1: Orchard Apple Counting

Executes:
  1. Dataset loading and verification (630 images, 25,440 apple annotations).
  2. Stage 1 Preprocessing across a representative batch (HSI decoupling, CLAHE, Spatial Denoising).
  3. High-Resolution 512x512 Tiling demo with 20% overlap.
  4. Generation of diagnostic outputs for report evaluation.
"""

import os
import sys

# Ensure src in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.minneapple_dataset import MinneAppleDataset, process_minneapple_batch
from src.stage1_preprocessing import preprocess_image, plot_stage1_diagnostics


def main():
    print("=" * 75)
    print("Machine Vision (UFMFRR-15-M) - Stage 1 Processing: MinneApple Dataset")
    print("=" * 75)

    dataset_dir = os.path.join(os.path.dirname(__file__), "minneapple-1-1", "train")
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] MinneApple dataset directory not found at: {dataset_dir}")
        print("Please run download_minneapple.py first.")
        return

    # 1. Load and Index Dataset
    dataset = MinneAppleDataset(dataset_dir=dataset_dir)

    # 2. Process a batch of 15 images through Stage 1
    summary = process_minneapple_batch(
        dataset=dataset,
        num_images=15,
        save_diagnostics=True,
        output_dir="output"
    )

    # 3. Demonstrate 512x512 Tiling with 20% overlap on a sample image
    print("\nDemonstrating High-Resolution 512x512 Tiling (20% Overlap) on Sample Image #1:")
    sample_img, anns, meta = dataset[0]
    print(f"  Full Image Dimensions: {sample_img.shape} (H, W, C)")
    
    tile_count = 0
    for patch, coords in MinneAppleDataset.tile_image(sample_img, patch_size=512, overlap=0.2):
        tile_count += 1
        # Preprocess each tile through Stage 1
        res_tile = preprocess_image(patch, mode="apple")
        if tile_count == 1:
            print(f"  First Tile Coords (y1, y2, x1, x2): {coords}, Patch Shape: {patch.shape}")
            print(f"  Stage 1 Preprocessed Tile Shape: {res_tile['preprocessed_rgb'].shape}")

    print(f"  -> Generated {tile_count} overlapping patches covering full 1280x720 canvas.")
    print("=" * 75)
    print("Stage 1 MinneApple pipeline execution completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
