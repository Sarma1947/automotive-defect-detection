# Automotive Surface Defect Detection

Anomaly detection in industrial components using ensemble deep learning on the MVTec Anomaly Detection dataset. Built as a portfolio project for Master thesis applications in the automotive industry (Bosch, Continental, BMW).

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![AUROC](https://img.shields.io/badge/Mean%20AUROC-90.24%25-green)

---

## Problem

Manufacturing quality control relies on manual visual inspection — slow, expensive, and error-prone. This project builds an automated defect detection system that trains only on normal images, requires no defect labels, and localizes exactly where defects occur.

---

## Results

| Model | Mean AUROC |
|---|---|
| ResNet50 baseline | 85.67% |
| DenseNet121 | 87.38% |
| EfficientNet-B4 | 88.83% |
| Ensemble ours | 90.24% |

Ensemble of ResNet50 + EfficientNet-B4 + DenseNet121 achieves 90.24% mean AUROC, outperforming all individual models by up to +4.57%.

![Ensemble Results](outputs/ensemble_comparison.png)

---

## Per-Category Results

| Category | ResNet50 | EfficientNet-B4 | DenseNet121 | Ensemble |
|---|---|---|---|---|
| bottle | 97.22 | 99.68 | 99.37 | 99.44 |
| cable | 87.52 | 81.07 | 90.09 | 90.67 |
| capsule | 83.61 | 77.94 | 85.20 | 84.76 |
| carpet | 83.83 | 97.43 | 87.56 | 93.22 |
| grid | 53.38 | 75.86 | 50.96 | 61.65 |
| hazelnut | 93.75 | 86.00 | 91.54 | 94.64 |
| leather | 99.73 | 100.00 | 98.27 | 100.00 |
| metal_nut | 75.56 | 89.39 | 80.84 | 86.80 |
| pill | 74.25 | 70.65 | 77.28 | 77.85 |
| screw | 74.01 | 79.59 | 81.94 | 86.10 |
| tile | 98.45 | 98.12 | 97.33 | 99.28 |
| toothbrush | 90.28 | 89.17 | 93.06 | 92.22 |
| transistor | 83.50 | 91.75 | 90.25 | 91.79 |
| wood | 93.68 | 99.04 | 94.21 | 97.81 |
| zipper | 96.27 | 96.74 | 92.75 | 97.43 |
| Mean | 85.67 | 88.83 | 87.38 | 90.24 |

---

## Defect Localization — Grad-CAM

Red regions show exactly where the model detects anomalies on the component surface.

![Grad-CAM](outputs/gradcam_visualization.png)

---

## Method

1. Extract deep features from normal training images using three pretrained CNN backbones
2. Build nearest-neighbor memory bank of normal features per category
3. At test time compute distance to nearest normal feature as anomaly score
4. Ensemble — normalize and average scores from all three models
5. Threshold score to classify Normal vs Defect
6. Grad-CAM highlights defect location on the image

---

## Dataset

MVTec Anomaly Detection Dataset — 15 industrial categories, 5000+ images, pixel-level annotations.

Download: https://www.mvtec.com/company/research/datasets/mvtec-ad

---

## Project Structure

- notebooks/01_eda.ipynb — EDA, experiments, full results
- src/dataset.py — PyTorch Dataset class
- src/model.py — Model definitions
- src/train.py — Training loop
- src/evaluate.py — Evaluation metrics
- outputs/ — Charts and visualizations
- app.py — Streamlit demo

---

## Setup

pip install -r requirements.txt

Run the demo:

streamlit run app.py

---

## Background

This project extends internship experience involving large-scale image classification (100,000+ images) using DenseNet121 with ensemble methods — applying the same methodology to industrial anomaly detection.

---

## Future Work

- Improve grid category (currently 61.65% AUROC) using texture-aware features
- Add pixel-level anomaly segmentation maps
- Optimize for real-time inference on automotive edge hardware
- Fine-tune on company-specific manufacturing datasets

