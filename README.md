# 🔭 DeepLense: Gravitational Lens Finding

This repository contains my evaluation task submissions for the **ML4SCI / DeepLense** organization for **Google Summer of Code (GSoC) 2026**. The primary focus of this project is the detection and classification of strong gravitational lenses using Deep Learning.

## 📁 Repository Structure

* `Common_Task.ipynb` / `common_task.py`: Solution for the mandatory DeepLense common evaluation task.
* `Specific_Test_V_Lens_Finding.ipynb`: Full training, validation, and inference pipeline for **Specific Test V (Lens Finding & Data Pipelines)**.
* `requirements.txt`: Python dependencies required to run the notebooks.

---

## 🚀 Specific Test V: Lens Finding & Data Pipelines

### Task Overview
The objective is to build a robust binary classification model to distinguish lensed galaxies from non-lensed galaxies using 3-channel (multi-filter) 64x64 observational images.

**The core challenge:** Extreme class imbalance. 
* **Training Set:** 1,730 lenses vs. 28,675 non-lenses (~1:16 ratio)
* **Test Set:** 195 lenses vs. 19,455 non-lenses (~1:100 ratio)

### Methodology & Architecture

To tackle the physical constraints and the severe class imbalance, the pipeline utilizes the following strategies:

1.  **Model Architecture (EfficientNet-B0):** Instead of a shallow custom CNN, I utilized a pretrained `EfficientNet-B0` backbone. To preserve the morphological features of the lenses (arcs, Einstein rings) and match the network's expected receptive field, images are upscaled to 128x128 during the transform pipeline.
2.  **Handling Imbalance (Focal Loss):**
    Standard Cross-Entropy fails at a 1:100 test imbalance. The model is trained using **Focal Loss** to dynamically down-weight easy negative examples (obvious non-lenses) and heavily penalize missed lenses. 
    
    $$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
    
    *Configuration: $\alpha = 0.95$ (heavily weighting the positive class) and $\gamma = 2.0$.*
3.  **Physical Symmetry Augmentations:**
    Astronomical images exhibit $D_4$ symmetry (no preferred "up" or "down" in space). The dataset is augmented using exact 90-degree rotations and flips to synthetically expand the rare lens class without interpolative blurring.
4.  **Robust Evaluation (5-Fold Stratified CV):**
    To prevent high variance from a single validation split, the pipeline utilizes 5-Fold Stratified Cross-Validation, ensuring stable out-of-fold (OOF) threshold tuning and evaluation.
5.  **Test-Time Augmentation (TTA):**
    At inference, the ensemble averages predictions over 8 deterministic physical states (4 rotations x 2 flips) to reduce predictive variance on borderline cases.

### Evaluation Metrics
Because accuracy is a misleading metric for imbalanced data, this model is evaluated primarily on its ranking and precision-recall capabilities:
* **ROC-AUC** (Receiver Operating Characteristic - Area Under Curve)
* **PR-AUC** (Precision-Recall Area Under Curve)

---

## ⚙️ How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/DeepLense-GSoC-2026.git](https://github.com/yourusername/DeepLense-GSoC-2026.git)
   cd DeepLense-GSoC-2026
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Setup:**
   * Download the dataset provided in the DeepLense task description.
   * Extract the files and update the `BASE_PATH` variable in the configuration block of `Specific_Test_V_Lens_Finding.ipynb` to point to your local directories (`train_lenses`, `train_nonlenses`, etc.).

4. **Execution:**
   * The notebook is structured to run end-to-end. 
   * *Note for Kaggle/Colab users:* If you experience multiprocessing `AssertionError` crashes during data loading, ensure `NUM_WORKERS = 0` is set in the configuration block.
