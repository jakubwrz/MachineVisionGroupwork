# Machine Vision (UFMFRR-15-M) - Group Project

## Repository Structure

- **`Jakub/`**: Stage 1 Preprocessing implementation by Jakub Wrzyszcz.
  - `stage1_preprocessing.py`: Standalone script for Stage 1 (Colour Space Decoupling, Intensity CLAHE, Median Denoising).
  - `Stage1_Preprocessing.ipynb`: Google Colab notebook for Stage 1.
  - `download_minneapple.py`: Dataset downloader using Roboflow SDK.

---

## Stage 1: Radiometric & Spatial Preprocessing (Units 2, 4, 5)

### Running Jakub's Stage 1 Code

#### In Google Colab:
```python
!git clone https://github.com/jakubwrz/MachineVisionGroupwork.git
%cd MachineVisionGroupwork/Jakub
!python stage1_preprocessing.py
```

#### Locally:
```bash
python Jakub/stage1_preprocessing.py
```
