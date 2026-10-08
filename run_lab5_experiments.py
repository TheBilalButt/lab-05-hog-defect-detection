"""
LAB 05: HOG-Based Industrial Defect Detection and Classification
Course: Computer Vision / Industrial Automated Visual Inspection
Author: Bilal Butt (@TheBilalButt)
Dataset: Northeastern University (NEU) Surface Defect Benchmark Dataset
Platform: Python 3.10+ | OpenCV | Scikit-Image | Scikit-Learn

This module implements the complete industrial inspection pipeline:
  1. Dataset ingestion & multi-class / binary labeling
  2. Industrial Preprocessing (Grayscale, Resizing 128x128, Normalization)
  3. Histogram of Oriented Gradients (HOG) Feature Extraction & Gradient Field Visualization
  4. Machine Learning Classification: Support Vector Machine (SVM) vs Random Forest
  5. Classifier Benchmarking: Accuracy, Precision, Recall, F1-Score, Confusion Matrices
  6. Parameter Ablation: Cell Sizes (4x4, 8x8, 16x16) and Orientations (6, 9, 12)
  7. Environmental Robustness Testing: Brightness variations, Gaussian noise, Rotation, Blur
  8. Automated Quality Control (QC) Decision Gate (ACCEPT/REJECT with Confidence Scoring)
"""

import os
import sys
import json
import time
import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from skimage import exposure
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold

# Ensure directories exist
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Defect class metadata
CLASS_LABELS = {
    'crazing': 'Crazing (Network Microcracks)',
    'inclusion': 'Inclusion (Oxide/Slag Particles)',
    'patches': 'Patches (Surface Plate Flaws)',
    'pitted_surface': 'Pitted Surface (Cavity Clusters)',
    'rolled-in_scale': 'Rolled-in Scale (Oxide Scale Pressing)',
    'scratches': 'Scratches (Abrasive Longitudinal Grooves)',
    'normal': 'Normal / Defect-Free Surface'
}

CLASSES = ['normal', 'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


class IndustrialInspectionEngine:
    def __init__(self, data_dir="data", target_size=(128, 128)):
        self.data_dir = data_dir
        self.target_size = target_size
        self.raw_images = []
        self.preprocessed_images = []
        self.binary_labels = []       # 0: Normal, 1: Defective
        self.multiclass_labels = []   # 0 to 6
        self.class_names = []
        self.filenames = []

    def load_dataset(self):
        """Task 1 & 2: Load raw industrial images, resize, and convert to grayscale."""
        image_paths = sorted(glob.glob(os.path.join(self.data_dir, "*.jpg")))
        if not image_paths:
            raise FileNotFoundError(f"No .jpg images found in {self.data_dir}")

        for path in image_paths:
            fname = os.path.basename(path)
            cat = fname.rsplit('_', 1)[0]
            if cat not in CLASS_TO_IDX:
                continue

            bgr = cv2.imread(path)
            if bgr is None:
                continue

            # Task 3: Grayscale conversion
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # Task 2: Standardized resolution resizing
            resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)

            self.filenames.append(fname)
            self.raw_images.append(resized)
            self.preprocessed_images.append(resized)
            self.class_names.append(cat)
            self.multiclass_labels.append(CLASS_TO_IDX[cat])
            self.binary_labels.append(0 if cat == 'normal' else 1)

        self.raw_images = np.array(self.raw_images)
        self.preprocessed_images = np.array(self.preprocessed_images)
        self.binary_labels = np.array(self.binary_labels)
        self.multiclass_labels = np.array(self.multiclass_labels)
        
        print(f"Loaded {len(self.filenames)} samples across {len(set(self.class_names))} categories.")
        print(f"Binary distribution: {np.sum(self.binary_labels == 0)} Normal, {np.sum(self.binary_labels == 1)} Defective.")
        return self

    def visualize_dataset_overview(self):
        """Generate publication-ready overview montage of all industrial surface categories."""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.patch.set_facecolor('#0f172a')
        
        sample_indices = {}
        for idx, cat in enumerate(self.class_names):
            if cat not in sample_indices:
                sample_indices[cat] = idx

        for i, cat in enumerate(CLASSES):
            ax = axes[i // 4, i % 4]
            ax.set_facecolor('#1e293b')
            idx = sample_indices[cat]
            img = self.preprocessed_images[idx]
            ax.imshow(img, cmap='gray')
            title_color = '#10b981' if cat == 'normal' else '#f43f5e'
            tag = "[NON-DEFECTIVE]" if cat == 'normal' else "[DEFECTIVE]"
            ax.set_title(f"{CLASS_LABELS[cat]}\n{tag}", fontsize=11, fontweight='bold', color=title_color, pad=10)
            ax.axis('off')

        # Empty 8th slot: Summary stats infographic
        ax_summary = axes[1, 3]
        ax_summary.set_facecolor('#1e293b')
        ax_summary.axis('off')
        summary_text = (
            "NEU SURFACE DEFECT DATASET\n"
            "---------------------------\n"
            f"Total Inspected: {len(self.raw_images)} Units\n"
            f"Image Resolution: {self.target_size[0]}x{self.target_size[1]} px\n"
            f"Color Format: Monochromatic 8-bit\n"
            f"Normal Class: 10 Samples (14.3%)\n"
            f"Defective Classes: 60 Samples (85.7%)\n"
            "Inspection Mode: Binary + 7-Class\n"
            "---------------------------\n"
            "Factory Line: Automated Rolling Mill"
        )
        ax_summary.text(0.5, 0.5, summary_text, color='#94a3b8', fontsize=10, family='monospace',
                        ha='center', va='center', bbox=dict(boxstyle='round,pad=1', facecolor='#0f172a', edgecolor='#334155'))

        plt.suptitle("Figure 1: Industrial Surface Defect Dataset - Representative Morphology", 
                     fontsize=15, fontweight='bold', color='#f8fafc', y=0.98)
        plt.tight_layout()
        plt.savefig("figures/figure1_industrial_dataset_inspection.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        print("Generated: figures/figure1_industrial_dataset_inspection.png")

    def extract_hog_features(self, images, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """Task 4: Extract HOG feature vectors."""
        features = []
        for img in images:
            feat = hog(
                img,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm='L2-Hys',
                visualize=False,
                feature_vector=True
            )
            features.append(feat)
        return np.array(features)

    def visualize_hog_representations(self):
        """Task 5: Side-by-side visualization of raw surfaces and HOG gradient orientation fields."""
        selected_classes = ['normal', 'crazing', 'pitted_surface', 'scratches']
        fig, axes = plt.subplots(4, 3, figsize=(14, 16))
        fig.patch.set_facecolor('#0b0f19')

        for row_idx, cat in enumerate(selected_classes):
            img_idx = next(i for i, c in enumerate(self.class_names) if c == cat)
            img = self.preprocessed_images[img_idx]

            # Compute HOG with visualization
            _, hog_vis = hog(
                img,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualize=True
            )
            hog_vis_rescaled = exposure.rescale_intensity(hog_vis, in_range=(0, 10))

            # Compute Sobel edge magnitude for structural comparison
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Subplot 1: Raw Surface
            axes[row_idx, 0].imshow(img, cmap='gray')
            axes[row_idx, 0].set_title(f"{CLASS_LABELS[cat]}\n[Original Surface]", color='#f8fafc', fontsize=11, fontweight='bold')
            axes[row_idx, 0].axis('off')

            # Subplot 2: Sobel Edge Magnitude
            axes[row_idx, 1].imshow(mag_norm, cmap='inferno')
            axes[row_idx, 1].set_title(f"{cat.capitalize()} Edges\n[Sobel Magnitude]", color='#38bdf8', fontsize=11, fontweight='bold')
            axes[row_idx, 1].axis('off')

            # Subplot 3: HOG Feature Field
            axes[row_idx, 2].imshow(hog_vis_rescaled, cmap='magma')
            axes[row_idx, 2].set_title(f"HOG Orientation Descriptors\n[Cells 8x8 | 9 Ori]", color='#a855f7', fontsize=11, fontweight='bold')
            axes[row_idx, 2].axis('off')

        plt.suptitle("Figure 2: HOG Descriptor Representations & Gradient Orientation Signatures", 
                     fontsize=15, fontweight='bold', color='#f8fafc', y=0.99)
        plt.tight_layout()
        plt.savefig("figures/figure3_hog_feature_maps.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        print("Generated: figures/figure3_hog_feature_maps.png")


class ClassifierBenchmarkingSuite:
    def __init__(self, X, y_binary, y_multi, class_names):
        self.X = X
        self.y_binary = y_binary
        self.y_multi = y_multi
        self.class_names = class_names
        self.results = {}

    def run_classifier_comparison(self):
        """Tasks 6, 7, 8, 9, 10: Train SVM and Random Forest, compare metrics, plot confusion matrices."""
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        classifiers = {
            'Support Vector Machine (Linear SVM)': SVC(kernel='linear', C=1.0, probability=True, random_state=42),
            'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        }

        comparison = {}
        confusion_matrices = {}
        classification_reports = {}

        for name, clf in classifiers.items():
            start_t = time.time()
            fold_acc, fold_prec, fold_rec, fold_f1 = [], [], [], []
            all_preds, all_trues = [], []

            for train_idx, test_idx in skf.split(self.X, self.y_binary):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y_binary[train_idx], self.y_binary[test_idx]

                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)

                fold_acc.append(accuracy_score(y_test, preds))
                fold_prec.append(precision_score(y_test, preds, zero_division=0))
                fold_rec.append(recall_score(y_test, preds, zero_division=0))
                fold_f1.append(f1_score(y_test, preds, zero_division=0))

                all_preds.extend(preds)
                all_trues.extend(y_test)

            total_time = time.time() - start_t
            cm = confusion_matrix(all_trues, all_preds)
            cr = classification_report(all_trues, all_preds, target_names=['Normal', 'Defective'], output_dict=True)

            comparison[name] = {
                'accuracy': float(np.mean(fold_acc)),
                'accuracy_std': float(np.std(fold_acc)),
                'precision': float(np.mean(fold_prec)),
                'recall': float(np.mean(fold_rec)),
                'f1_score': float(np.mean(fold_f1)),
                'inference_time_ms': float((total_time / len(self.X)) * 1000)
            }
            confusion_matrices[name] = cm
            classification_reports[name] = cr

        self.results = comparison
        self.plot_confusion_matrices(confusion_matrices)
        
        with open("results/classifier_performance_metrics.json", "w") as f:
            json.dump({
                "comparison": comparison,
                "classification_reports": classification_reports
            }, f, indent=4)
        print("Generated: results/classifier_performance_metrics.json")
        return comparison

    def plot_confusion_matrices(self, cm_dict):
        """Plot side-by-side normalized confusion matrices for SVM and Random Forest."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('#0f172a')

        colors = ['Blues', 'Purples']
        for idx, (name, cm) in enumerate(cm_dict.items()):
            ax = axes[idx]
            ax.set_facecolor('#1e293b')
            
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap=colors[idx],
                        xticklabels=['Normal', 'Defective'],
                        yticklabels=['Normal', 'Defective'],
                        cbar=False, ax=ax, annot_kws={"size": 13, "weight": "bold"})

            ax.set_title(f"{name}\nAcc: {self.results[name]['accuracy']*100:.1f}% | F1: {self.results[name]['f1_score']*100:.1f}%",
                         fontsize=12, fontweight='bold', color='#f8fafc', pad=12)
            ax.set_xlabel("Predicted Inspection Decision", color='#94a3b8', fontsize=11, labelpad=8)
            ax.set_ylabel("True Ground Truth", color='#94a3b8', fontsize=11, labelpad=8)
            ax.tick_params(colors='#cbd5e1')

        plt.suptitle("Figure 3: Classifier Performance & Normalized Confusion Matrix Benchmarking",
                     fontsize=15, fontweight='bold', color='#f8fafc', y=1.02)
        plt.tight_layout()
        plt.savefig("figures/figure4_classifier_confusion_matrices.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        print("Generated: figures/figure4_classifier_confusion_matrices.png")


class HOGParameterAblationStudy:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def run_ablation(self):
        """Task 11: Investigate effect of HOG cell sizes (4x4, 8x8, 16x16) and orientations (6, 9, 12)."""
        cell_sizes = [(4, 4), (8, 8), (16, 16)]
        orientations_list = [6, 9, 12]

        ablation_results = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        print("\n--- Running HOG Parameter Ablation Study ---")
        for cell in cell_sizes:
            for ori in orientations_list:
                t0 = time.time()
                features = []
                for img in self.images:
                    feat = hog(
                        img,
                        orientations=ori,
                        pixels_per_cell=cell,
                        cells_per_block=(2, 2),
                        block_norm='L2-Hys',
                        visualize=False,
                        feature_vector=True
                    )
                    features.append(feat)
                X_feat = np.array(features)
                feat_time = (time.time() - t0) / len(self.images)

                clf = SVC(kernel='linear', C=1.0, random_state=42)
                f1_scores = []
                acc_scores = []
                for tr_idx, te_idx in skf.split(X_feat, self.labels):
                    clf.fit(X_feat[tr_idx], self.labels[tr_idx])
                    p = clf.predict(X_feat[te_idx])
                    f1_scores.append(f1_score(self.labels[te_idx], p, zero_division=0))
                    acc_scores.append(accuracy_score(self.labels[te_idx], p))

                record = {
                    'cell_size': f"{cell[0]}x{cell[1]}",
                    'orientations': ori,
                    'feature_dim': int(X_feat.shape[1]),
                    'extraction_ms': float(feat_time * 1000),
                    'accuracy': float(np.mean(acc_scores)),
                    'f1_score': float(np.mean(f1_scores))
                }
                ablation_results.append(record)
                print(f"Cell: {record['cell_size']:>5} | Ori: {ori:>2} | Dim: {record['feature_dim']:>5} | Ext: {record['extraction_ms']:.2f}ms | Acc: {record['accuracy']*100:.1f}% | F1: {record['f1_score']*100:.1f}%")

        with open("results/hog_parameter_ablation.json", "w") as f:
            json.dump(ablation_results, f, indent=4)

        self.plot_ablation(ablation_results)
        return ablation_results

    def plot_ablation(self, results):
        """Plot parameter sensitivity charts across cell sizes and orientations."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor('#0f172a')

        # Subplot 1: Cell Size Impact (Orientations fixed at 9)
        ax1 = axes[0]
        ax1.set_facecolor('#1e293b')
        fixed_ori_9 = [r for r in results if r['orientations'] == 9]
        cell_labels = [r['cell_size'] for r in fixed_ori_9]
        accs = [r['accuracy'] * 100 for r in fixed_ori_9]
        f1s = [r['f1_score'] * 100 for r in fixed_ori_9]
        dims = [r['feature_dim'] for r in fixed_ori_9]

        x = np.arange(len(cell_labels))
        width = 0.35
        ax1.bar(x - width/2, accs, width, label='Accuracy (%)', color='#38bdf8', edgecolor='#0284c7')
        ax1.bar(x + width/2, f1s, width, label='F1-Score (%)', color='#a855f7', edgecolor='#7e22ce')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{c}\n(Dim: {d})" for c, d in zip(cell_labels, dims)], color='#cbd5e1', fontsize=10)
        ax1.set_title("HOG Cell Size Analysis (Fixed 9 Orientations)", color='#f8fafc', fontsize=12, fontweight='bold', pad=10)
        ax1.set_ylabel("Performance (%)", color='#94a3b8', fontsize=11)
        ax1.set_ylim(70, 105)
        ax1.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc')
        ax1.grid(color='#334155', linestyle='--', alpha=0.5)

        # Subplot 2: Orientations Impact (Cell size fixed at 8x8)
        ax2 = axes[1]
        ax2.set_facecolor('#1e293b')
        fixed_cell_8 = [r for r in results if r['cell_size'] == '8x8']
        ori_labels = [f"{r['orientations']} Bins" for r in fixed_cell_8]
        accs_ori = [r['accuracy'] * 100 for r in fixed_cell_8]
        f1s_ori = [r['f1_score'] * 100 for r in fixed_cell_8]
        times = [r['extraction_ms'] for r in fixed_cell_8]

        x2 = np.arange(len(ori_labels))
        ax2.bar(x2 - width/2, accs_ori, width, label='Accuracy (%)', color='#10b981', edgecolor='#059669')
        ax2.bar(x2 + width/2, f1s_ori, width, label='F1-Score (%)', color='#f59e0b', edgecolor='#d97706')
        ax2.set_xticks(x2)
        ax2.set_xticklabels([f"{o}\n({t:.1f}ms)" for o, t in zip(ori_labels, times)], color='#cbd5e1', fontsize=10)
        ax2.set_title("HOG Orientation Bins Analysis (Fixed 8x8 Cell Size)", color='#f8fafc', fontsize=12, fontweight='bold', pad=10)
        ax2.set_ylabel("Performance (%)", color='#94a3b8', fontsize=11)
        ax2.set_ylim(70, 105)
        ax2.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc')
        ax2.grid(color='#334155', linestyle='--', alpha=0.5)

        plt.suptitle("Figure 4: HOG Parameter Ablation Study (Cell Sizes vs Orientations)",
                     fontsize=15, fontweight='bold', color='#f8fafc', y=1.02)
        plt.tight_layout()
        plt.savefig("figures/figure5_hog_parameter_ablation.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        print("Generated: figures/figure5_hog_parameter_ablation.png")


class RobustnessStressTester:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def apply_perturbation(self, img, condition):
        """Apply realistic environmental factory imaging corruptions."""
        h, w = img.shape
        if condition == 'brightness_low':
            return np.clip(img.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)
        elif condition == 'brightness_high':
            return np.clip(img.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
        elif condition == 'gaussian_noise':
            noise = np.random.normal(0, 20, img.shape)
            return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif condition == 'rotation':
            M = cv2.getRotationMatrix2D((w // 2, h // 2), 15, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif condition == 'blur':
            return cv2.GaussianBlur(img, (7, 7), 2.5)
        return img

    def evaluate_robustness(self):
        """Task 12: Test system robustness under brightness, noise, rotation, and blur."""
        np.random.seed(42)
        clean_features = []
        for img in self.images:
            f = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
            clean_features.append(f)
        X_clean = np.array(clean_features)

        clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
        clf.fit(X_clean, self.labels)

        base_preds = clf.predict(X_clean)
        baseline_acc = accuracy_score(self.labels, base_preds)
        baseline_f1 = f1_score(self.labels, base_preds)

        conditions = [
            ('Baseline (Clean)', 'clean'),
            ('Low Brightness (0.5x)', 'brightness_low'),
            ('High Brightness (1.5x)', 'brightness_high'),
            ('Gaussian Noise (sigma=20)', 'gaussian_noise'),
            ('Conveyor Rotation (+15 deg)', 'rotation'),
            ('Optical Blur (k=7, sigma=2.5)', 'blur')
        ]

        robustness_records = []
        perturbed_samples = []

        for label_name, cond in conditions:
            if cond == 'clean':
                robustness_records.append({
                    'condition': label_name,
                    'accuracy': float(baseline_acc),
                    'f1_score': float(baseline_f1),
                    'accuracy_drop': 0.0,
                    'f1_drop': 0.0
                })
                perturbed_samples.append(self.images[0])
                continue

            perturbed_imgs = [self.apply_perturbation(img, cond) for img in self.images]
            perturbed_samples.append(perturbed_imgs[0])
            
            p_features = []
            for p_img in perturbed_imgs:
                pf = hog(p_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
                p_features.append(pf)
            X_pert = np.array(p_features)

            preds = clf.predict(X_pert)
            acc = accuracy_score(self.labels, preds)
            f1 = f1_score(self.labels, preds)

            robustness_records.append({
                'condition': label_name,
                'accuracy': float(acc),
                'f1_score': float(f1),
                'accuracy_drop': float(baseline_acc - acc),
                'f1_drop': float(baseline_f1 - f1)
            })
            print(f"Condition: {label_name:<28} | Acc: {acc*100:.1f}% (Drop: -{(baseline_acc-acc)*100:.1f}%) | F1: {f1*100:.1f}%")

        with open("results/robustness_analysis.json", "w") as f:
            json.dump(robustness_records, f, indent=4)

        self.plot_robustness(robustness_records, conditions, perturbed_samples)
        return robustness_records

    def plot_robustness(self, records, conditions, samples):
        """Plot degradation curves and sample corrupted inspection frames."""
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('#0f172a')
        gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.2])

        for col_idx, ((name, cond), sample_img) in enumerate(zip(conditions, samples)):
            ax = fig.add_subplot(gs[0, col_idx])
            ax.set_facecolor('#1e293b')
            ax.imshow(sample_img, cmap='gray')
            ax.set_title(name.split(' (')[0], color='#38bdf8', fontsize=10, fontweight='bold', pad=8)
            ax.axis('off')

        ax_chart = fig.add_subplot(gs[1, :])
        ax_chart.set_facecolor('#1e293b')

        names = [r['condition'] for r in records]
        accs = [r['accuracy'] * 100 for r in records]
        f1s = [r['f1_score'] * 100 for r in records]
        x = np.arange(len(names))
        width = 0.35

        rects1 = ax_chart.bar(x - width/2, accs, width, label='Accuracy (%)', color='#38bdf8', edgecolor='#0284c7')
        rects2 = ax_chart.bar(x + width/2, f1s, width, label='F1-Score (%)', color='#f43f5e', edgecolor='#e11d48')

        ax_chart.set_ylabel('Performance (%)', color='#94a3b8', fontsize=12)
        ax_chart.set_title('Degradation Analysis Across Industrial Perturbation Modalities', color='#f8fafc', fontsize=13, fontweight='bold', pad=12)
        ax_chart.set_xticks(x)
        ax_chart.set_xticklabels(names, rotation=15, ha='right', color='#cbd5e1', fontsize=10)
        ax_chart.set_ylim(50, 105)
        ax_chart.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc', loc='lower left')
        ax_chart.grid(color='#334155', linestyle='--', alpha=0.5)

        for i in range(1, len(records)):
            drop_txt = f"-{records[i]['accuracy_drop']*100:.1f}%"
            ax_chart.annotate(drop_txt, xy=(x[i] - width/2, accs[i]), xytext=(0, 4),
                              textcoords="offset points", ha='center', color='#facc15', fontsize=9, fontweight='bold')

        plt.suptitle("Figure 5: Industrial Environmental Robustness & Perturbation Tolerance Evaluation",
                     fontsize=15, fontweight='bold', color='#f8fafc', y=0.98)
        plt.tight_layout()
        plt.savefig("figures/figure6_robustness_perturbation_analysis.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        print("Generated: figures/figure6_robustness_perturbation_analysis.png")


class QualityControlDecisionGate:
    def __init__(self, clf, target_size=(128, 128)):
        self.clf = clf
        self.target_size = target_size

    def inspect_product(self, img_input, threshold=0.50):
        """Task 13: Final Industrial Quality Control Decision Module."""
        if isinstance(img_input, str):
            img = cv2.imread(img_input, cv2.IMREAD_GRAYSCALE)
        else:
            img = img_input

        if img is None:
            raise ValueError("Invalid image input to QC Gate.")

        resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        feat = hog(resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys').reshape(1, -1)

        probs = self.clf.predict_proba(feat)[0]
        prob_defective = probs[1]
        prob_normal = probs[0]

        is_defective = prob_defective >= threshold
        prediction_str = "DEFECTIVE" if is_defective else "NON-DEFECTIVE"
        confidence = prob_defective if is_defective else prob_normal
        action_str = "REJECT PRODUCT" if is_defective else "ACCEPT PRODUCT"

        result = {
            "prediction": prediction_str,
            "confidence_percent": round(float(confidence * 100), 2),
            "action": action_str,
            "raw_probability_defective": float(prob_defective),
            "raw_probability_normal": float(prob_normal)
        }
        return result, resized

    def print_inspection_banner(self, result):
        """Format terminal output strictly matching factory specification."""
        print("\n" + "=" * 40)
        print("PRODUCT INSPECTION RESULT")
        print("=" * 40)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence_percent']}%")
        print(f"Action: {result['action']}")
        print("=" * 40)

    def generate_inspection_report_cards(self, test_samples):
        """Generate visual inspection certificates with HUD pass/fail stamps."""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.patch.set_facecolor('#0f172a')

        for idx, (img_path, true_label) in enumerate(test_samples[:8]):
            ax = axes[idx // 4, idx % 4]
            ax.set_facecolor('#1e293b')
            res, processed_img = self.inspect_product(img_path)

            ax.imshow(processed_img, cmap='gray')
            color = '#10b981' if res['action'] == 'ACCEPT PRODUCT' else '#f43f5e'
            banner = f"{res['action']}\nPred: {res['prediction']} ({res['confidence_percent']}%)"
            ax.set_title(banner, color=color, fontsize=10, fontweight='bold', pad=8)
            ax.axis('off')

            rect = plt.Rectangle((0, 0), processed_img.shape[1], processed_img.shape[0],
                                 linewidth=3, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        plt.suptitle("Figure 6: Automated Factory QC Inspection Decision Output & Pass/Reject Telemetry",
                     fontsize=15, fontweight='bold', color='#f8fafc', y=0.98)
        plt.tight_layout()
        plt.savefig("figures/figure7_qc_inspection_decision_samples.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        print("Generated: figures/figure7_qc_inspection_decision_samples.png")


def main():
    print("=" * 70)
    print("STARTING LAB 05: HOG-BASED INDUSTRIAL DEFECT CLASSIFICATION PIPELINE")
    print("Author: Bilal Butt (@TheBilalButt)")
    print("=" * 70)

    engine = IndustrialInspectionEngine(data_dir="data", target_size=(128, 128))
    engine.load_dataset()
    engine.visualize_dataset_overview()

    print("\n--- Extracting Default HOG Descriptors (8x8 cells, 9 bins) ---")
    X_hog = engine.extract_hog_features(engine.preprocessed_images)
    print(f"Extracted feature matrix shape: {X_hog.shape}")
    engine.visualize_hog_representations()

    benchmarker = ClassifierBenchmarkingSuite(X_hog, engine.binary_labels, engine.multiclass_labels, engine.class_names)
    metrics = benchmarker.run_classifier_comparison()

    ablation = HOGParameterAblationStudy(engine.preprocessed_images, engine.binary_labels)
    ablation.run_ablation()

    stress_tester = RobustnessStressTester(engine.preprocessed_images, engine.binary_labels)
    stress_tester.evaluate_robustness()

    svm_clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    svm_clf.fit(X_hog, engine.binary_labels)
    qc_gate = QualityControlDecisionGate(svm_clf, target_size=(128, 128))

    normal_sample = os.path.join("data", "normal_1.jpg")
    defective_sample = os.path.join("data", "scratches_1.jpg")
    
    print("\nEvaluating Sample 1 (Defect-Free Roll):")
    res1, _ = qc_gate.inspect_product(normal_sample)
    qc_gate.print_inspection_banner(res1)

    print("\nEvaluating Sample 2 (Defective Scratched Roll):")
    res2, _ = qc_gate.inspect_product(defective_sample)
    qc_gate.print_inspection_banner(res2)

    test_paths = [(os.path.join("data", f), c) for f, c in zip(engine.filenames, engine.class_names)]
    qc_gate.generate_inspection_report_cards(test_paths)

    print("\n" + "=" * 70)
    print("LAB 05 EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("All figures saved to figures/ and metrics saved to results/.")
    print("=" * 70)


if __name__ == '__main__':
    main()
