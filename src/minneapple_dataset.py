"""
MinneApple Dataset Loader & Stage 1 Preprocessing Integration
Machine Vision (UFMFRR-15-M) Group Project

Primary Contributor: [Group Member Name / Student ID]
Task 1: Orchard Apple Counting Pipeline

Technical Overview:
-------------------
This module provides the official dataset interface for the MinneApple orchard dataset,
integrating directly with Stage 1 (Radiometric & Spatial Preprocessing):
  1. COCO-format Annotation Parser:
     - Extracts image metadata, bounding boxes, categories, and ground-truth apple counts.
  2. High-Resolution Tiling & Patch Extraction:
     - Implements overlapping 512x512 window slicing (20% stride overlap) to prevent
       downsampling loss of small, distant apples in canopy clusters.
  3. Batch Stage 1 Preprocessing:
     - Decouples chrominance/luminance via HSI conversion.
     - Performs CLAHE contrast normalisation on the Intensity channel.
     - Applies non-linear 3x3 median and Gaussian spatial filtering.
     - Logs quantitative contrast and dynamic range metrics across the dataset.
"""

import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Generator
import numpy as np
import cv2
import matplotlib.pyplot as plt

from .stage1_preprocessing import (
    preprocess_image,
    plot_stage1_diagnostics,
    DEFAULT_APPLE_CONFIG,
    Stage1Preprocessor,
)


class MinneAppleDataset:
    """
    COCO-format dataset loader for the MinneApple orchard benchmark.
    """

    def __init__(self,
                 dataset_dir: str = "minneapple-1-1/train",
                 annotation_file: Optional[str] = None):
        """
        Initialises the MinneApple dataset loader.

        Parameters
        ----------
        dataset_dir : str
            Directory containing train/validation images and COCO JSON.
        annotation_file : str, optional
            Path to the COCO JSON annotations. If None, looks for '_annotations.coco.json'.
        """
        self.dataset_dir = os.path.abspath(dataset_dir)
        if annotation_file is None:
            self.annotation_path = os.path.join(self.dataset_dir, "_annotations.coco.json")
        else:
            self.annotation_path = os.path.abspath(annotation_file)

        if not os.path.exists(self.annotation_path):
            raise FileNotFoundError(f"Annotations file not found at: {self.annotation_path}")

        print(f"Loading MinneApple annotations from: {self.annotation_path}")
        with open(self.annotation_path, "r", encoding="utf-8") as f:
            self.coco_data = json.load(f)

        self._build_indices()

    def _build_indices(self):
        """Builds fast lookup indices for images and annotations."""
        # Map category_id -> category info
        self.categories = {cat["id"]: cat["name"] for cat in self.coco_data.get("categories", [])}

        # Map image_id -> image info
        self.images = {img["id"]: img for img in self.coco_data.get("images", [])}
        self.image_ids = list(self.images.keys())

        # Map image_id -> list of annotations
        self.annotations_by_image: Dict[int, List[Dict[str, Any]]] = {img_id: [] for img_id in self.image_ids}
        for ann in self.coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id in self.annotations_by_image:
                self.annotations_by_image[img_id].append(ann)

        print(f"Dataset indexed: {len(self.image_ids)} images, "
              f"{len(self.coco_data.get('annotations', []))} apple annotations.")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieves image, annotations, and metadata by index.

        Returns
        -------
        img_rgb : np.ndarray
            RGB image (H, W, 3) uint8.
        annotations : list
            List of bounding box dicts for the image.
        meta : dict
            Image metadata (id, file_name, width, height, apple_count).
        """
        img_id = self.image_ids[idx]
        img_meta = self.images[img_id].copy()
        file_name = img_meta["file_name"]
        img_path = os.path.join(self.dataset_dir, file_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file does not exist: {img_path}")

        bgr = cv2.imread(img_path)
        if bgr is None:
            raise ValueError(f"Failed to read image at: {img_path}")
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        annotations = self.annotations_by_image.get(img_id, [])
        img_meta["apple_count"] = len(annotations)
        img_meta["file_path"] = img_path

        return img_rgb, annotations, img_meta

    def get_image_by_filename(self, file_name: str) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """Retrieves image by filename."""
        for idx, img_id in enumerate(self.image_ids):
            if self.images[img_id]["file_name"] == file_name:
                return self.__getitem__(idx)
        raise KeyError(f"Image with filename {file_name} not found.")

    @staticmethod
    def tile_image(image: np.ndarray,
                   patch_size: int = 512,
                   overlap: float = 0.2) -> Generator[Tuple[np.ndarray, Tuple[int, int, int, int]], None, None]:
        """
        Slices a high-resolution image into overlapping square patches
        (e.g., 512x512 with 20% overlap) as specified in the project blueprint.

        Parameters
        ----------
        image : np.ndarray
            Input image of shape (H, W, C).
        patch_size : int
            Height and width of each tile.
        overlap : float
            Fractional overlap between adjacent tiles (e.g. 0.2 for 20%).

        Yields
        ------
        patch : np.ndarray
            Extracted patch of shape (patch_size, patch_size, C).
        coords : tuple
            (y1, y2, x1, x2) bounding coordinates within the full image.
        """
        h, w = image.shape[:2]
        stride = int(patch_size * (1.0 - overlap))

        y_steps = list(range(0, max(1, h - patch_size + 1), stride))
        if len(y_steps) == 0 or (y_steps[-1] + patch_size < h):
            y_steps.append(max(0, h - patch_size))

        x_steps = list(range(0, max(1, w - patch_size + 1), stride))
        if len(x_steps) == 0 or (x_steps[-1] + patch_size < w):
            x_steps.append(max(0, w - patch_size))

        for y in y_steps:
            for x in x_steps:
                y1 = y
                y2 = min(h, y + patch_size)
                x1 = x
                x2 = min(w, x + patch_size)

                # Pad if patch is smaller than patch_size at edge
                patch = image[y1:y2, x1:x2]
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    padded = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
                    padded[:patch.shape[0], :patch.shape[1]] = patch
                    patch = padded

                yield patch, (y1, y2, x1, x2)


def process_minneapple_batch(dataset: MinneAppleDataset,
                            num_images: int = 20,
                            config: Optional[Dict[str, Any]] = None,
                            save_diagnostics: bool = True,
                            output_dir: str = "output") -> Dict[str, Any]:
    """
    Executes Stage 1 Preprocessing across a batch of MinneApple images,
    logging radiometric improvement metrics and saving diagnostic figures.

    Parameters
    ----------
    dataset : MinneAppleDataset
        Initialised MinneApple dataset instance.
    num_images : int
        Number of images to process in the batch (default: 20).
    config : dict, optional
        Custom Stage 1 config dictionary (defaults to DEFAULT_APPLE_CONFIG).
    save_diagnostics : bool
        Whether to save representative diagnostic plots.
    output_dir : str
        Directory to save figures into.

    Returns
    -------
    summary : dict
        Aggregated metrics (mean latency, contrast expansion ratio, total apples).
    """
    preprocessor = Stage1Preprocessor(config=config, mode="apple")
    os.makedirs(output_dir, exist_ok=True)

    latencies_ms = []
    contrast_gains = []
    total_apples = 0

    print(f"\nProcessing {min(num_images, len(dataset))} images through Stage 1 Preprocessing...")
    print("-" * 75)
    print(f"{'Idx':<4} | {'Image Filename':<36} | {'Apples':<6} | {'Raw Std':<8} | {'Eq Std':<8} | {'Time (ms)':<8}")
    print("-" * 75)

    num_to_run = min(num_images, len(dataset))
    for i in range(num_to_run):
        img_rgb, anns, meta = dataset[i]
        apple_count = meta["apple_count"]
        total_apples += apple_count

        t0 = time.perf_counter()
        res = preprocessor.process(img_rgb)
        t_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(t_ms)

        raw_std = float(res["channel_3_raw"].std())
        eq_std = float(res["channel_3_denoised"].std())
        contrast_gain = eq_std / max(1.0, raw_std)
        contrast_gains.append(contrast_gain)

        fname_short = meta["file_name"][:34] + ".." if len(meta["file_name"]) > 36 else meta["file_name"]
        print(f"{i+1:<4} | {fname_short:<36} | {apple_count:<6} | {raw_std:<8.2f} | {eq_std:<8.2f} | {t_ms:<8.1f}")

        # Save first 2 representative diagnostic figures (e.g. for report)
        if save_diagnostics and i < 2:
            out_path = os.path.join(output_dir, f"figure_stage1_minneapple_sample_{i+1}.png")
            fig = plot_stage1_diagnostics(
                res,
                title=f"MinneApple Sample #{i+1} ({apple_count} Apples) - Stage 1 Preprocessing",
                save_path=out_path
            )
            plt.close(fig)

    mean_time = float(np.mean(latencies_ms))
    mean_gain = float(np.mean(contrast_gains))

    print("-" * 75)
    print(f"Batch Summary:")
    print(f"  - Total Images Processed: {num_to_run}")
    print(f"  - Total Apple Instances:  {total_apples}")
    print(f"  - Mean Latency per Image: {mean_time:.2f} ms ({1000.0/mean_time:.1f} FPS)")
    print(f"  - Mean Contrast Gain:     {mean_gain:.2f}x dynamic range expansion")
    print("-" * 75)

    return {
        "images_processed": num_to_run,
        "total_apples": total_apples,
        "latencies_ms": latencies_ms,
        "mean_latency_ms": mean_time,
        "contrast_gains": contrast_gains,
        "mean_contrast_gain": mean_gain,
    }
