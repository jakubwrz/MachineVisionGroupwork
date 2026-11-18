"""
MinneApple Dataset Downloader via Roboflow SDK
Machine Vision (UFMFRR-15-M) Group Project

Primary Contributor: Jakub Wrzyszcz
"""

import os
from roboflow import Roboflow

def download_dataset():
    api_key = os.environ.get("ROBOFLOW_API_KEY", "YOUR_ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("apple-e1dtv").project("minneapple-1")
    version = project.version(1)
    dataset = version.download("coco")
    print(f"Dataset downloaded to: {dataset.location}")

if __name__ == "__main__":
    download_dataset()
