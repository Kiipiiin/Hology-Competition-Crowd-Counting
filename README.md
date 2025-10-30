# Crowd Counting using CSRNet + Patch-Based Density Estimation  
**Deep Learning Project — Density Map Regression & Patch-Based Training**

This project implements a crowd counting system using a modified CSRNet architecture combined with a patch-based training pipeline.  
Instead of detecting individuals, the model predicts a **density map**, and the total crowd count is obtained by summing all pixel values.

---

## Dataset
Link: https://drive.google.com/drive/folders/1tX3uTlIGFQHDUel0ftASEqMBXfeKIEko?usp=drive_link

## Key Features

1. Patch-based dataset generation (splitting each large image into smaller patches).
2. Adaptive augmentation based on crowd density to reduce imbalance and improve model robustness.
3. Automatic HDF5 (`.h5`) density map generation from annotation points stored in `.json`.
4. Two model variants included in this project:
   - **CSRNet-Lite** (best performance, multi-scale receptive field)
   - **CSRNet-Tiny** (smaller model, lower accuracy)

The main evaluation metric used is **MAE — Mean Absolute Error**.

---

## Architecture Overview

### CSRNet-Lite (BEST MODEL)

This version keeps the CSRNet idea of using **dilated convolutions**, which allows the receptive field to grow without additional pooling.  
This is crucial for density map generation because:

- Spatial resolution remains intact (no excessive downsampling),
- Dilated convolutions capture both local texture and global scene context,
- Works effectively across scenes with sparse and extremely dense populations.

### CSRNet-Tiny

Smaller version of the model with reduced dilated convolution layers.  
It is lighter, but:

- The receptive field is not wide enough to capture crowd context,
- Performs poorly in images where density varies drastically.

> CSRNet-Lite consistently produces lower MAE on both validation and test sets.

---

## Dataset Pipeline

1. Read human head points from `.json` annotations.
2. Convert annotation points into **Gaussian density maps** and save them as `.h5`.
3. Split the full resolution image into patches (`9 × 9` configurable).
4. Apply safe augmentation (rotation, color jitter).
5. If a patch contains more than 10 heads, it has a chance to be duplicated and augmented to avoid dataset dominance by empty/sparse patches.

Patch duplication probabilities:

- Low-density patch → lower duplication probability,
- High-density patch → higher duplication probability.

This prevents the model from becoming biased toward empty or low-crowd patches.

---

## Data Characteristics

This project was trained on a **limited dataset**:

- Training images: 1520  
- Validation images: 190  
- Test images: 190  

The dataset is **highly imbalanced**:

- 1219 images contain fewer than 200 people  
- 205 images contain between 200–500 people  
- Only 96 images contain more than 500 people  

Due to the large portion of sparse images, patch duplication and density-aware augmentation were necessary to avoid underfitting on dense crowd scenarios.

---

## Test Results (Final Model: CSRNet-Lite)

- Test set evaluated: **190 images**
- Mean Absolute Error (MAE): **33.47**

This means on average the model’s predicted crowd count differs from ground truth by about **±34 people per image**, which is considered strong performance given the dataset imbalance and limited training data.

---


