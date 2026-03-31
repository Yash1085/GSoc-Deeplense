# 🔭 GSoC 2026: ML4SCI DeepLense — Gravitational Lens Finding

[![Organization](https://img.shields.io/badge/Organization-ML4SCI-blue.svg)](https://ml4sci.org/)
[![Project](https://img.shields.io/badge/Project-DeepLense-purple.svg)](https://ml4sci.org/deeplense/)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](#)

This repository contains the solution and complete data pipeline for **Specific Test V: Lens Finding & Data Pipelines**, submitted as part of the Google Summer of Code (GSoC) 2026 evaluation phase for the DeepLense project.

**Author:** Yash Yadav  
**Kaggle Profile:** [Yash2072005](https://www.kaggle.com/yash2072005)  
**LinkedIn:** [yash-yadav007](https://www.linkedin.com/in/yash-yadav007/)

---

## 📑 Project Overview

The objective of this task is to develop a robust binary classification model capable of distinguishing strong gravitational lenses from non-lensed galaxies using 3-channel (multi-filter) $64 \times 64$ observational images. 

The primary challenge of this dataset is the **extreme class imbalance**, mirroring real-world wide-field surveys (e.g., HSC-SSP):
* **Training Set Imbalance:** 1 : 16.6 (Lenses to Non-lenses)
* **Test Set Imbalance:** 1 : 100 (Lenses to Non-lenses)

Because standard ROC-AUC becomes highly insensitive to False Positives under severe imbalance, this pipeline is strictly optimized and evaluated using **PR-AUC (Precision-Recall AUC)** and **TPR @ 1% FPR**.

---

## 🧠 Methodology & Engineering Highlights

To combat domain-specific challenges, the following advanced machine learning techniques were implemented:

### 1. Mathematical Imbalance Handling
* **Macro-Balancing:** Implemented PyTorch's `WeightedRandomSampler` to ensure the model sees a balanced 1:1 ratio during batch generation.
* **Micro-Hard Example Mining:** Engineered a **Custom Alpha-Stripped Focal Loss**. By removing the $\alpha$ weighting parameter (since the batch is already macro-balanced by the sampler) and keeping the $\gamma$ parameter, the gradients are forced to focus strictly on the faintest, most difficult Einstein rings without double-penalizing the majority class.

### 2. Physics-Aware Modeling
* **Primary Backbone (EfficientNet-B0):** Utilized for its optimal compound scaling and native Squeeze-and-Excitation (SE) blocks, which dynamically recalibrate channel-wise features to focus on extended lensing arcs.
* **Continuous $SO(2)$ Equivariance ($C_8$-ENN):** Implemented a baseline Equivariant Neural Network using the `escnn` library (`gspaces.rot2dOnR2(N=8)`). Unlike standard CNNs that rely on dataset bloat via geometric augmentations, this model has the continuous rotational symmetry of Einstein rings structurally baked into its filters.

### 3. Rigorous Validation & Inference
* **5-Fold Stratified Cross-Validation:** Ensured model stability and generated reliable Out-Of-Fold (OOF) metrics.
* **Max-Pooled Test-Time Augmentation (TTA):** Deployed a 5-pass TTA using `np.max` aggregation rather than the mean. This preserves anomaly signals (faint lenses) that might only be highly visible in specific orientations.
* **Ensemble Inference:** Final test predictions are generated via an ensemble average of the 5-Fold models.

### 4. Interpretability
* **Grad-CAM:** Applied Gradient-weighted Class Activation Mapping to True Positives, False Positives, and False Negatives to visually prove the network is focusing on morphological lensing features (arcs/rings) rather than central deflector artifacts.

---

## 📊 Results Summary

*(Note: Replace with your exact final metric outputs from the notebook)*

| Model Pipeline | ROC-AUC | PR-AUC | TPR @ 1% FPR |
| :--- | :---: | :---: | :---: |
| **EfficientNet-B0 (5-Fold Ensemble + TTA)** | `0.99xx` | `0.9xxx` | `0.9xxx` |
| **$C_8$ Equivariant Neural Network Baseline** | `0.9xxx` | `0.8xxx` | `0.8xxx` |

---

## ⚙️ Reproducibility & Installation

The entire pipeline is contained within a single, modular Jupyter Notebook (`gsoc-deeplense-test5-final.ipynb`) designed to run on Kaggle or a local GPU environment.

**Dependencies:**
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib tqdm
pip install escnn        # For the C8 Equivariant Neural Network
pip install grad-cam     # For Interpretability visualizations
