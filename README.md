# Machine Vision (UFMFRR-15-M) Group Project
## Stage 1: Radiometric & Spatial Preprocessing (Units 2, 4, 5)

This repository contains the complete implementation for **Stage 1** of the unified 5-stage machine vision pipeline for both:
1. **Task 1: Orchard Apple Counting** (Handling non-uniform solar canopy illumination gradients and dappled shadowing).
2. **Task 2: Multi-Resistor Colour Band Classification** (Suppressing cylindrical specular reflection streaks and compensating for bench lighting).

---

## 1. Unified Architecture & Mathematical Foundations

### 1.1 Colour Space Conversion (Unit 2)
Raw RGB imaging entangles surface reflectance (chrominance) with illumination geometry and intensity (luminance). To decouple colour from illumination:
- Input RGB image is converted to **HSI (Hue, Saturation, Intensity)** using the exact Gonzalez & Woods equations:
  $$I = \frac{R + G + B}{3}$$
  $$S = 1 - \frac{3}{R + G + B} \min(R, G, B)$$
  $$\theta = \arccos\left( \frac{0.5 \cdot ((R - G) + (R - B))}{\sqrt{(R - G)^2 + (R - B)(G - B)} + \epsilon} \right)$$
  $$H = \begin{cases} \theta & \text{if } B \le G \\ 2\pi - \theta & \text{if } B > G \end{cases}$$
- **Key Insight:** Modifying intensity ($I$) leaves chromaticity ($H, S$) unchanged, avoiding artificial hue shifts during contrast equalisation.

### 1.2 Contrast Normalisation (Unit 4)
- **Cumulative Distribution Function (CDF) Histogram Equalisation:**
  $$s_k = T(r_k) = (L - 1) \sum_{j=0}^{k} p_r(r_j) = \frac{L - 1}{MN} \sum_{j=0}^{k} n_j$$
- **CLAHE (Contrast Limited Adaptive Histogram Equalisation):** Applied exclusively to the Intensity ($I$) channel to enhance shadowed canopy foliage and resistor bodies without blowing out background noise.

### 1.3 Spatial Denoising (Unit 5)
- **Median Filter (Rank-Order):** A $3 \times 3$ (Apples) or $5 \times 5$ (Resistors) non-linear median filter completely suppresses narrow specular reflection spikes and impulse noise without blurring edges (unlike a linear averaging filter which causes edge bleeding).
- **Gaussian Kernel:** A $3 \times 3$ Gaussian filter ($\sigma = 0.8 - 1.0$) attenuates high-frequency sensor noise.

---

## 2. Directory Structure

```text
MachineVision_Groupwork/
├── src/
│   ├── __init__.py
│   └── stage1_preprocessing.py   # Core reusable Stage 1 module
├── notebooks/
│   └── Stage1_Radiometric_and_Spatial_Preprocessing.ipynb # Google Colab notebook
├── tests/
│   └── test_stage1.py             # Rigorous unit test suite (built-in unittest)
├── output/
│   ├── figure_stage1_apple_diagnostics.png
│   ├── figure_stage1_resistor_diagnostics.png
│   └── figure_stage1_specular_profile.png
├── run_stage1_demo.py             # Standalone demonstration & figure generator
├── generate_notebook.py           # Colab notebook generation script
└── README.md
```

---

## 3. Quick Start & Execution

### Running Unit Tests
```bash
python -m unittest tests/test_stage1.py
```

### Running the Demonstration Script
```bash
python run_stage1_demo.py
```
This processes both benchmark scenarios and saves publication-ready figures to `output/`.

### Using the Python Module Directly
```python
import cv2
from src.stage1_preprocessing import preprocess_image, plot_stage1_diagnostics

# Load RGB image
img_rgb = cv2.cvtColor(cv2.imread("sample.jpg"), cv2.COLOR_BGR2RGB)

# Task 1: Orchard Apple Preprocessing
apple_out = preprocess_image(img_rgb, mode="apple")

# Task 2: Multi-Resistor Preprocessing
resistor_out = preprocess_image(img_rgb, mode="resistor")

# Plot side-by-side diagnostic figures
fig = plot_stage1_diagnostics(resistor_out)
```

---

## 4. Assessment Compliance & Colab Requirements
- **Cell Headers:** Every cell in `notebooks/Stage1_Radiometric_and_Spatial_Preprocessing.ipynb` contains the mandatory `# Primary Contributor: [Name]` header.
- **Diagnostic Plots:** Side-by-side diagnostic figures display RGB inputs, channel breakdowns, equalised outputs, and empirical CDF linearisation.
- **Unified Pipeline:** Both tasks share the same core functions and classes, adhering to the 5-stage architecture.
