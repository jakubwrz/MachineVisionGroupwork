"""
MinneApple Dataset Downloader via Roboflow SDK
Machine Vision (UFMFRR-15-M) Group Project - Task 1

Primary Contributor: [Name]
"""

import os
import sys
from roboflow import Roboflow

def download_dataset(api_key: str = None, target_format: str = "coco"):
    """
    Downloads the MinneApple dataset from Roboflow in the specified format.

    Parameters
    ----------
    api_key : str, optional
        Roboflow API key. If None, reads from ROBOFLOW_API_KEY environment variable
        or falls back to placeholder.
    target_format : str
        Dataset format ('coco', 'yolov8', etc.). Defaults to 'coco'.
    """
    key = api_key or os.environ.get("ROBOFLOW_API_KEY", "YOUR_ROBOFLOW_API_KEY")
    if key == "YOUR_ROBOFLOW_API_KEY":
        print("[WARNING] Using placeholder API key. Please set your ROBOFLOW_API_KEY before running.")
        print("Example: export ROBOFLOW_API_KEY='your_key' or pass api_key='your_key'")

    print("Connecting to Roboflow API...")
    rf = Roboflow(api_key=key)

    print("Accessing workspace 'apple-e1dtv' and project 'minneapple-1'...")
    project = rf.workspace("apple-e1dtv").project("minneapple-1")

    print("Selecting version 1...")
    version = project.version(1)

    print(f"Downloading dataset in '{target_format}' format...")
    dataset = version.download(target_format)

    print(f"\nDataset successfully downloaded to: {dataset.location}")
    return dataset


if __name__ == "__main__":
    download_dataset()
