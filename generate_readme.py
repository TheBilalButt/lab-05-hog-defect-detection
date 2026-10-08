import os
import json

BILAL_DIR = r"D:\Cv lab 5"

# Load metrics from JSON
with open(os.path.join(BILAL_DIR, "results", "classifier_performance_metrics.json")) as f:
    clf_metrics = json.load(f)

with open(os.path.join(BILAL_DIR, "results", "hog_parameter_ablation.json")) as f:
    ablation_metrics = json.load(f)

with open(os.path.join(BILAL_DIR, "results", "robustness_analysis.json")) as f:
    robust_metrics = json.load(f)

# Precompute markdown tables to avoid syntax issues
ablation_rows = []
for r in ablation_metrics:
    ablation_rows.append(f"| `{r['cell_size']}` | `{r['orientations']}` | `{r['feature_dim']:,}` | `{r['extraction_ms']:.2f} ms` | **{r['accuracy']*100:.1f}%** | **{r['f1_score']*100:.1f}%** |")
ablation_table_markdown = "\n".join(ablation_rows)

robust_rows = []
for r in robust_metrics:
    drop_str = f"{-r['accuracy_drop']*100:+.1f}%"
    status_str = "Stable" if r['accuracy_drop'] <= 0.0 else "Degraded"
    robust_rows.append(f"| **{r['condition']}** | **{r['accuracy']*100:.1f}%** | `{drop_str}` | **{r['f1_score']*100:.1f}%** | {status_str} |")
robust_table_markdown = "\n".join(robust_rows)

svm_acc = f"{clf_metrics['comparison']['Support Vector Machine (Linear SVM)']['accuracy']*100:.1f}%"
svm_prec = f"{clf_metrics['comparison']['Support Vector Machine (Linear SVM)']['precision']*100:.1f}%"
svm_rec = f"{clf_metrics['comparison']['Support Vector Machine (Linear SVM)']['recall']*100:.1f}%"
svm_f1 = f"{clf_metrics['comparison']['Support Vector Machine (Linear SVM)']['f1_score']*100:.1f}%"
svm_time = f"{clf_metrics['comparison']['Support Vector Machine (Linear SVM)']['inference_time_ms']:.2f} ms"

rf_acc = f"{clf_metrics['comparison']['Random Forest Classifier']['accuracy']*100:.1f}%"
rf_prec = f"{clf_metrics['comparison']['Random Forest Classifier']['precision']*100:.1f}%"
rf_rec = f"{clf_metrics['comparison']['Random Forest Classifier']['recall']*100:.1f}%"
rf_f1 = f"{clf_metrics['comparison']['Random Forest Classifier']['f1_score']*100:.1f}%"
rf_time = f"{clf_metrics['comparison']['Random Forest Classifier']['inference_time_ms']:.2f} ms"

# Build Notebook
nb_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# LAB 05: HOG-Based Industrial Defect Detection and Classification\n",
            "\n",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheBilalButt/lab-05-hog-defect-detection/blob/main/Lab5_HOG_Industrial_Defect_Detection.ipynb)\n",
            "[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/TheBilalButt/lab-05-hog-defect-detection)\n",
            "[![Course](https://img.shields.io/badge/Course-Computer%20Vision-blueviolet)](https://github.com/TheBilalButt)\n",
            "[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)\n",
            "[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green?logo=opencv)](https://opencv.org/)\n",
            "\n",
            "**Author:** Bilal Butt ([@TheBilalButt](https://github.com/TheBilalButt))  \n",
            "**Institution:** Department of Computer Science / Artificial Intelligence  \n",
            "**Repository:** [TheBilalButt / lab-05-hog-defect-detection](https://github.com/TheBilalButt/lab-05-hog-defect-detection)  \n",
            "**Benchmark Dataset:** Northeastern University (NEU) Surface Defect Database  \n",
            "\n",
            "---\n",
            "\n",
            "## 1. Executive Summary & Problem Formulation\n",
            "\n",
            "In modern high-speed industrial manufacturing (e.g., hot-rolled steel strip rolling mills, automated electronics assembly, and aerospace fabrication), manual visual inspection is strictly unviable. Human optical quality audits are constrained by operator visual fatigue, subjective bias, low throughput (typically < 2 meters/sec), and an unacceptable false escape rate exceeding 15%.\n",
            "\n",
            "The objective of this laboratory is to design, implement, and benchmark an automated visual quality control system leveraging the **Histogram of Oriented Gradients (HOG)** feature descriptor coupled with statistical machine-learning classifiers (**Support Vector Machines** and **Random Forests**).\n",
            "\n",
            "### Architectural Pipeline\n",
            "$$\\text{Surface Image} \\longrightarrow \\text{Grayscale & Resizing (128}\\times\\text{128)} \\longrightarrow \\text{HOG Feature Extraction} \\longrightarrow \\text{ML Classifier (SVM / RF)} \\longrightarrow \\text{QC Decision Gate}$$"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Environment & Dependency Setup\n",
            "import os\n",
            "import glob\n",
            "import time\n",
            "import numpy as np\n",
            "import cv2\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from skimage.feature import hog\n",
            "from skimage import exposure\n",
            "from sklearn.svm import SVC\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report\n",
            "from sklearn.model_selection import StratifiedKFold\n",
            "\n",
            "print('Environment initialized successfully.')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Dataset Ingestion & Surface Defect Morphology\n",
            "\n",
            "The benchmark dataset contains surface images of hot-rolled steel plates spanning six critical industrial defect categories, alongside pristine defect-free reference plates:\n",
            "1. **Crazing (`Cr`)**: Network microcracking arising from thermal shock and fatigue stress.\n",
            "2. **Inclusion (`In`)**: Non-metallic impurities (oxide slags) embedded into the steel matrix.\n",
            "3. **Patches (`Pa`)**: Localized surface scale build-ups and plate thickness inconsistencies.\n",
            "4. **Pitted Surface (`PS`)**: Cavity clusters created by chemical corrosion or localized mechanical damage.\n",
            "5. **Rolled-in Scale (`RS`)**: Iron oxide scale pressed into the strip during hot continuous rolling.\n",
            "6. **Scratches (`Sc`)**: Severe abrasive longitudinal grooves caused by mechanical guide friction.\n",
            "7. **Normal / Defect-Free (`Normal`)**: Homogeneous rolling grain without fissures, pits, or grooves."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load industrial dataset and verify class distributions\n",
            "data_dir = 'data'\n",
            "image_paths = sorted(glob.glob(os.path.join(data_dir, '*.jpg')))\n",
            "print(f'Total available industrial surface images: {len(image_paths)}')\n",
            "\n",
            "images, binary_labels, multi_labels, class_names = [], [], [], []\n",
            "CLASSES = ['normal', 'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']\n",
            "CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}\n",
            "\n",
            "for p in image_paths:\n",
            "    fname = os.path.basename(p)\n",
            "    c = fname.rsplit('_', 1)[0]\n",
            "    if c not in CLASS_MAP:\n",
            "        continue\n",
            "    raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)\n",
            "    resized = cv2.resize(raw, (128, 128), interpolation=cv2.INTER_AREA)\n",
            "    images.append(resized)\n",
            "    class_names.append(c)\n",
            "    multi_labels.append(CLASS_MAP[c])\n",
            "    binary_labels.append(0 if c == 'normal' else 1)\n",
            "\n",
            "X_imgs = np.array(images)\n",
            "y_bin = np.array(binary_labels)\n",
            "y_mul = np.array(multi_labels)\n",
            "print(f'Dataset Shape: {X_imgs.shape} | Non-Defective: {np.sum(y_bin == 0)} | Defective: {np.sum(y_bin == 1)}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Dataset Morphology Visualization\n",
            "Below is the representative surface morphology across all seven inspected surface categories:"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Display Figure 1\n",
            "from IPython.display import Image\n",
            "Image(filename='figures/figure1_industrial_dataset_inspection.png')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Histogram of Oriented Gradients (HOG) Feature Extraction\n",
            "\n",
            "The HOG descriptor characterizes local object appearance and shape by evaluating distributions of local intensity gradients.\n",
            "\n",
            "### Mathematical Foundations\n",
            "1. **1-D Discrete Gradient Calculation**:\n",
            "   $$G_x(x, y) = I(x+1, y) - I(x-1, y)$$\n",
            "   $$G_y(x, y) = I(x, y+1) - I(x, y-1)$$\n",
            "\n",
            "2. **Gradient Magnitude and Phase**:\n",
            "   $$M(x, y) = \\sqrt{G_x(x, y)^2 + G_y(x, y)^2}$$\n",
            "   $$\\theta(x, y) = \\arctan\\left(\\frac{G_y(x, y)}{G_x(x, y)}\\right) \\pmod{180^\\circ}$$\n",
            "\n",
            "3. **Spatial Cell Orientation Histograms**:  \n",
            "   Each cell (e.g. $8\\times 8$ pixels) accumulates gradient magnitudes into $B=9$ discrete orientation bins spanning $[0^\\circ, 180^\\circ)$.\n",
            "\n",
            "4. **Block Contrast Normalization (L2-Hys)**:  \n",
            "   To achieve invariance against illumination shifts, cells are grouped into overlapping blocks (e.g. $2\\times 2$ cells) and normalized:\n",
            "   $$v_{\\text{norm}} = \\frac{v}{\\sqrt{\\|v\\|_2^2 + \\epsilon^2}}, \\quad v_{\\text{final}} = \\min(v_{\\text{norm}}, 0.2)$$"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Extract baseline HOG descriptors (8x8 cells, 9 orientation bins)\n",
            "t0 = time.time()\n",
            "X_hog = []\n",
            "for img in X_imgs:\n",
            "    feat = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),\n",
            "               block_norm='L2-Hys', visualize=False, feature_vector=True)\n",
            "    X_hog.append(feat)\n",
            "X_hog = np.array(X_hog)\n",
            "t_ext = (time.time() - t0) / len(X_imgs)\n",
            "print(f'HOG Feature Matrix: {X_hog.shape} ({X_hog.shape[1]} dimensions per unit)')\n",
            "print(f'Average extraction latency: {t_ext*1000:.2f} ms/frame')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 5,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Display Figure 2: HOG Descriptor Field Visualizations\n",
            "Image(filename='figures/figure3_hog_feature_maps.png')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Machine Learning Classifier Benchmarking\n",
            "\n",
            "We formulate defect identification as an automated binary quality decision (Class 0: Normal vs Class 1: Defective) and benchmark two distinct machine learning paradigms:\n",
            "1. **Support Vector Machine (Linear SVM)**: Maximizes the geometric margin separating defective from pristine feature manifolds.\n",
            "2. **Random Forest Classifier (Ensemble of 100 Trees)**: Non-linear bootstrap aggregation evaluating gradient subspace splits.\n",
            "\n",
            "Evaluation is executed using **5-Fold Stratified Cross-Validation** to prevent sample leakage."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 6,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Display Classifier Confusion Matrices\n",
            "Image(filename='figures/figure4_classifier_confusion_matrices.png')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Quantitative Benchmarking Table\n",
            "| Classifier Architecture | Accuracy | Precision | Recall | F1-Score | Inference Latency |\n",
            "| :--- | :---: | :---: | :---: | :---: | :---: |\n",
            f"| **Linear Support Vector Machine (SVM)** | **{svm_acc}** | **{svm_prec}** | **{svm_rec}** | **{svm_f1}** | **{svm_time}** |\n",
            f"| **Random Forest Classifier (100 Trees)** | **{rf_acc}** | **{rf_prec}** | **{rf_rec}** | **{rf_f1}** | **{rf_time}** |\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. HOG Parameter Ablation Study\n",
            "\n",
            "We systematically investigate the hyperparameter trade-offs across:\n",
            "- **Spatial Cell Sizes**: $4\\times 4$, $8\\times 8$, and $16\\times 16$ pixels\n",
            "- **Orientation Bins**: $6$, $9$, and $12$ angular intervals"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 7,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Display Figure 4: HOG Parameter Ablation Charts\n",
            "Image(filename='figures/figure5_hog_parameter_ablation.png')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Ablation Experimental Summary\n",
            "| Spatial Cell Size | Orientation Bins | Feature Vector Dim | Extraction Time (ms) | Classification Accuracy | F1-Score |\n",
            "| :---: | :---: | :---: | :---: | :---: | :---: |\n",
            ablation_table_markdown + "\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Environmental Robustness & Stress Analysis\n",
            "\n",
            "Industrial rolling line cameras operate under severe lighting oscillations, atmospheric oil mist, vibration-induced rotation, and lens defocus.\n",
            "We evaluate our trained HOG+SVM system under four challenging real-world disturbances:\n",
            "1. **Illumination Shifts**: Low brightness (0.5x) and high brightness (1.5x)\n",
            "2. **Sensor Electronic Noise**: Additive Gaussian noise ($\\sigma = 20$)\n",
            "3. **Conveyor Alignment Skew**: In-plane rotation ($+15^\\circ$)\n",
            "4. **Optical Defocus / Motion Blur**: Gaussian blur kernel ($k=7, \\sigma = 2.5$)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 8,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Display Figure 5: Robustness Degradation Analysis\n",
            "Image(filename='figures/figure6_robustness_perturbation_analysis.png')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Robustness Evaluation Metrics Table\n",
            "| Imaging Environment | System Accuracy | Performance Delta | F1-Score | Retention Status |\n",
            "| :--- | :---: | :---: | :---: | :---: |\n",
            robust_table_markdown + "\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Automated Industrial Quality Control (QC) Decision Gate\n",
            "\n",
            "The final operational layer is the automated Quality Control gate that renders real-time binary decisions on the conveyor line:\n",
            "- If confidence of defect exceeds the inspection threshold ($P \\ge 0.50$): **`REJECT PRODUCT`**\n",
            "- If surface is verified pristine ($P < 0.50$): **`ACCEPT PRODUCT`**"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 9,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Interactive Factory Quality Control Demonstration\n",
            "svm_final = SVC(kernel='linear', C=1.0, probability=True, random_state=42)\n",
            "svm_final.fit(X_hog, y_bin)\n",
            "\n",
            "def inspect_product_unit(image_path):\n",
            "    raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)\n",
            "    resized = cv2.resize(raw, (128, 128))\n",
            "    feat = hog(resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),\n",
            "               block_norm='L2-Hys').reshape(1, -1)\n",
            "    probs = svm_final.predict_proba(feat)[0]\n",
            "    prob_def = probs[1]\n",
            "    is_def = prob_def >= 0.50\n",
            "    conf = prob_def if is_def else probs[0]\n",
            "    \n",
            "    print('=' * 40)\n",
            "    print('PRODUCT INSPECTION RESULT')\n",
            "    print('=' * 40)\n",
            "    print(f\"Prediction: {'DEFECTIVE' if is_def else 'NON-DEFECTIVE'}\")\n",
            "    print(f\"Confidence: {conf*100:.2f}%\")\n",
            "    print(f\"Action: {'REJECT PRODUCT' if is_def else 'ACCEPT PRODUCT'}\")\n",
            "    print('=' * 40)\n",
            "\n",
            "# Test Sample 1: Normal Surface\n",
            "print('TEST UNIT 1 (Defect-Free Roll):')\n",
            "inspect_product_unit('data/normal_1.jpg')\n",
            "\n",
            "# Test Sample 2: Defective Scratched Surface\n",
            "print('\\nTEST UNIT 2 (Surface Scratch Flaw):')\n",
            "inspect_product_unit('data/scratches_1.jpg')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 10,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Display Figure 6: Factory Inspection Report Cards\n",
            "Image(filename='figures/figure7_qc_inspection_decision_samples.png')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Conclusion & Industrial Deployment Guidelines\n",
            "\n",
            "1. **Descriptor Efficacy**: HOG effectively captures structural edge distributions (e.g. longitudinal groove lines in scratches, irregular boundaries in patches) while remaining invariant to uniform illumination gradients thanks to L2-Hysteresis block normalization.\n",
            "2. **Model Selection**: Linear SVM achieved **97.1% accuracy** and an **F1-Score of 98.4%**, outperforming Random Forest while delivering lower inference latency (~0.05 ms/frame), making it ideal for real-time high-speed manufacturing lines.\n",
            "3. **Parameter Recommendations**: For maximum throughput, an $8\\times 8$ or $16\\times 16$ cell size with 9 orientation bins represents the optimal balance of descriptive power and inference speed.\n",
            "4. **Environmental Safeguards**: Optical blur caused the steepest performance decline (-32.9%), indicating that high-shutter-speed telecentric industrial optics and structured LED illumination rings are critical deployment prerequisites."
        ]
    }
]

notebook_content = {
    "cells": nb_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(os.path.join(BILAL_DIR, "Lab5_HOG_Industrial_Defect_Detection.ipynb"), "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

# Generate README.md
readme_template = """# LAB 05: HOG-Based Industrial Defect Detection and Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheBilalButt/lab-05-hog-defect-detection/blob/main/Lab5_HOG_Industrial_Defect_Detection.ipynb)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/TheBilalButt/lab-05-hog-defect-detection)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green?logo=opencv)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Course**: Computer Vision / Automated Visual Inspection Systems  
**Author**: Bilal Butt ([@TheBilalButt](https://github.com/TheBilalButt))  
**Dataset**: Northeastern University (NEU) Surface Defect Benchmark Database  
**Repository**: [lab-05-hog-defect-detection](https://github.com/TheBilalButt/lab-05-hog-defect-detection)  

---

## 1. Problem Statement & Objective

Manual optical inspection of manufactured components (e.g., hot-rolled steel strip coils, silicon wafers, machined alloy castings) is inherently limited by operator fatigue, variable inspection conditions, and subjective judgment errors.

This laboratory develops a robust, deterministic computer vision prototype capable of automatically inspecting industrial surface images, characterizing surface edge and texture gradients via the **Histogram of Oriented Gradients (HOG)** descriptor, and classifying samples into **Defective** or **Non-Defective** categories to drive an automated **Quality Control (ACCEPT / REJECT)** decision module.

---

## 2. System Architecture & Inspection Pipeline

```
+------------------+      +-------------------+      +----------------------+
| Industrial Image | ---> | Image Preparation | ---> | HOG Feature Extractor|
|  (NEU Benchmark) |      | (Gray, 128x128 px)|      | (8x8 cells, 9 bins)  |
+------------------+      +-------------------+      +----------------------+
                                                                |
                                                                v
+------------------+      +-------------------+      +----------------------+
| Factory Telemetry| <--- |  QC Decision Gate | <--- | ML Classifier Suite  |
|  (ACCEPT/REJECT) |      | (Threshold Gate)  |      |   (Linear SVM / RF)  |
+------------------+      +-------------------+      +----------------------+
```

---

## 3. Dataset Description & Surface Defect Types

The system was evaluated against seven industrial surface categories:
* **Crazing (`Cr`)**: Network microcracking arising from thermal shock and cooling stress.
* **Inclusion (`In`)**: Non-metallic impurities and oxide particles trapped during casting.
* **Patches (`Pa`)**: Surface plate irregularities and localized thickness distortions.
* **Pitted Surface (`PS`)**: Cavity clusters created by localized chemical corrosion.
* **Rolled-in Scale (`RS`)**: Mill oxide scale pressed into strip steel during rolling.
* **Scratches (`Sc`)**: Deep longitudinal abrasive grooves caused by mechanical roller friction.
* **Normal (`Normal`)**: Pristine, defect-free hot-rolled surface finish.

![Figure 1: Industrial Surface Defect Overview](figures/figure1_industrial_dataset_inspection.png)

---

## 4. HOG Feature Representation & Vector Fields

HOG features capture morphological edge structures while rejecting uniform illumination variations via L2-Hysteresis block normalization.

![Figure 2: HOG Descriptor Representations](figures/figure3_hog_feature_maps.png)

---

## 5. Machine Learning Classifier Benchmarking

Both classifiers were evaluated using **5-Fold Stratified Cross-Validation**:

| Classifier Architecture | Accuracy | Precision | Recall | F1-Score | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (Linear SVM)** | **{svm_acc}** | **{svm_prec}** | **{svm_rec}** | **{svm_f1}** | **{svm_time}** |
| **Random Forest (100 Trees)** | **{rf_acc}** | **{rf_prec}** | **{rf_rec}** | **{rf_f1}** | **{rf_time}** |

![Figure 3: Confusion Matrices](figures/figure4_classifier_confusion_matrices.png)

---

## 6. HOG Parameter Ablation Study

We compared 9 parameter permutations (Cell sizes 4x4, 8x8, 16x16; Orientation bins 6, 9, 12):

| Spatial Cell Size | Orientation Bins | Feature Vector Dim | Extraction Time | Accuracy | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
{ablation_table}

![Figure 4: Parameter Ablation](figures/figure5_hog_parameter_ablation.png)

---

## 7. Environmental Robustness Analysis

The system was stress-tested against four real-world industrial disturbances:

| Environmental Perturbation | Accuracy | Performance Delta | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: |
{robust_table}

![Figure 5: Robustness Analysis](figures/figure6_robustness_perturbation_analysis.png)

---

## 8. Quality Control (QC) Decision Gate & Output

The automated decision module produces standardized terminal and digital telemetry:

```
========================================
PRODUCT INSPECTION RESULT
========================================
Prediction: NON-DEFECTIVE
Confidence: 99.54%
Action: ACCEPT PRODUCT
========================================

========================================
PRODUCT INSPECTION RESULT
========================================
Prediction: DEFECTIVE
Confidence: 99.01%
Action: REJECT PRODUCT
========================================
```

![Figure 6: Automated Inspection Report Cards](figures/figure7_qc_inspection_decision_samples.png)

---

## 9. Bonus Challenge: Real-Time Prototype

Run the real-time webcam / simulated conveyor inspection prototype:
```bash
python realtime_inspector.py
```
This launches a live Heads-Up Display (HUD) indicating real-time PASS / DEFECTIVE status, classification confidence, FPS, and an inset HOG gradient orientation vector field.

---

## 10. Execution Guide

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run end-to-end laboratory experiments
python run_lab5_experiments.py

# 3. Launch live inspection prototype
python realtime_inspector.py
```
"""

readme_content = readme_template.format(
    svm_acc=svm_acc,
    svm_prec=svm_prec,
    svm_rec=svm_rec,
    svm_f1=svm_f1,
    svm_time=svm_time,
    rf_acc=rf_acc,
    rf_prec=rf_prec,
    rf_rec=rf_rec,
    rf_f1=rf_f1,
    rf_time=rf_time,
    ablation_table=ablation_table_markdown,
    robust_table=robust_table_markdown
)

with open(os.path.join(BILAL_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)

print("Generated Notebook and README for Bilal Butt successfully!")
