# Machine Vision (UFMFRR-15-M) - Stage 1 Preprocessing

This repository contains the concise, standalone implementation for **Stage 1 (Radiometric & Spatial Preprocessing)**.

---

## 1. What Stage 1 Does (Units 2, 4, 5)
- **Colour Space Conversion (Unit 2):** Converts input RGB into HSV/HSI to decouple intrinsic chrominance (Hue & Saturation) from illumination (Value/Intensity).
- **Contrast Normalisation (Unit 4):** Applies Contrast-Limited Adaptive Histogram Equalisation (**CLAHE**) exclusively to the Intensity channel to normalise harsh canopy shadows without causing hue shifts.
- **Spatial Denoising (Unit 5):** Applies a $3 \times 3$ (apples) or $5 \times 5$ (resistors) **Median Filter** to remove speckle noise and specular glare streaks without blurring edges, followed by a $3 \times 3$ **Gaussian Filter**.

---

## 2. File Overview

- `stage1_preprocessing.py`: Single standalone script that performs Stage 1 on your dataset and displays the before/after results.
- `Stage1_Preprocessing.ipynb`: Single-cell Google Colab notebook.
- `download_minneapple.py`: Script to download the MinneApple dataset via the Roboflow SDK.

---

## 3. How to Run

### In Google Colab
```python
!git clone https://github.com/jakubwrz/MachineVisionGroupwork.git
%cd MachineVisionGroupwork
!python stage1_preprocessing.py
```

Or open `Stage1_Preprocessing.ipynb` directly in Google Colab.

### Locally
```bash
python stage1_preprocessing.py
```
