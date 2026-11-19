import os
import glob
import shutil
import cv2
from processing import enhance_resistor_crop

RAW_DATASET_DIR = "resistor_dataset_raw"          # Unzipped Roboflow export
ENHANCED_DATASET_DIR = "resistor_dataset_enhanced"

# 1. Make a fresh copy so re-running never double-enhances the images
if os.path.exists(ENHANCED_DATASET_DIR):
    shutil.rmtree(ENHANCED_DATASET_DIR)
shutil.copytree(RAW_DATASET_DIR, ENHANCED_DATASET_DIR)

# 2. Apply enhancement to every image across train/, valid/, and test/
exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
img_paths = []
for ext in exts:
    img_paths.extend(glob.glob(os.path.join(ENHANCED_DATASET_DIR, "*", "images", ext)))

for path in img_paths:
    img = cv2.imread(path)
    if img is not None:
        cv2.imwrite(path, enhance_resistor_crop(img))

print(f"Done: enhanced {len(img_paths)} images -> {ENHANCED_DATASET_DIR}/")
