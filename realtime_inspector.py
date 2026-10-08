"""
LAB 05 BONUS CHALLENGE: Real-Time Industrial Defect Inspection Prototype
Author: Bilal Butt (@TheBilalButt)
Course: Computer Vision

This script streams video frames from a connected camera / factory conveyor feed
(or loops simulated industrial surface feed if webcam unavailable), extracts HOG
features in real time, classifies surface integrity via SVM, and displays an
industrial Heads-Up Display (HUD) overlay:
  - Green PASS / ACCEPT banner if non-defective
  - Red DEFECTIVE / REJECT banner if defective
  - Real-time confidence gauge & inspection FPS
  - Inset HOG gradient vector visualization
"""

import cv2
import time
import glob
import os
import numpy as np
from skimage.feature import hog
from skimage import exposure
from sklearn.svm import SVC

def train_baseline_model():
    data_dir = "data"
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    if not image_paths:
        print("Data directory empty or missing.")
        return None

    X, y = [], []
    for path in image_paths:
        cat = os.path.basename(path).rsplit('_', 1)[0]
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        resized = cv2.resize(gray, (128, 128))
        feat = hog(resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        X.append(feat)
        y.append(0 if cat == 'normal' else 1)

    clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    clf.fit(X, y)
    print("Baseline SVM model trained on industrial dataset.")
    return clf

def main():
    clf = train_baseline_model()
    if clf is None:
        return

    # Attempt to open hardware webcam
    cap = cv2.VideoCapture(0)
    use_webcam = cap.isOpened()
    
    # If webcam not available, stream simulated surface images from dataset
    simulated_images = sorted(glob.glob(os.path.join("data", "*.jpg")))
    sim_idx = 0

    print("\n=======================================================")
    print("STARTING REAL-TIME INDUSTRIAL SURFACE INSPECTOR (HUD)")
    print("Mode:", "HARDWARE CAMERA FEED" if use_webcam else "SIMULATED CONVEYOR STREAM")
    print("Press 'q' to stop inspection prototype.")
    print("=======================================================\n")

    frame_count = 0
    start_time = time.time()
    fps = 0.0

    while True:
        if use_webcam:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crop = cv2.resize(gray, (128, 128))
            display = frame.copy()
        else:
            time.sleep(0.08)  # simulate ~12 FPS inspection conveyor speed
            sim_path = simulated_images[sim_idx % len(simulated_images)]
            sim_idx += 1
            raw = cv2.imread(sim_path)
            display = cv2.resize(raw, (640, 480))
            gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
            crop = cv2.resize(gray, (128, 128))

        # HOG Extraction & Visualization
        feat, hog_vis = hog(crop, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                            block_norm='L2-Hys', visualize=True)
        feat = feat.reshape(1, -1)

        # Classification
        probs = clf.predict_proba(feat)[0]
        prob_defective = probs[1]
        is_defective = prob_defective >= 0.50
        confidence = prob_defective if is_defective else probs[0]

        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        # Render Industrial Heads-Up Display (HUD)
        h, w, _ = display.shape
        overlay = display.copy()

        # Top Banner
        status_color = (0, 30, 200) if is_defective else (0, 180, 50)  # BGR
        status_text = "STATUS: REJECT [DEFECTIVE]" if is_defective else "STATUS: ACCEPT [PASS]"
        
        cv2.rectangle(overlay, (0, 0), (w, 65), (20, 24, 33), -1)
        cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

        cv2.rectangle(display, (0, 0), (w, 65), status_color, 2)
        cv2.putText(display, status_text, (20, 42), cv2.FONT_HERSHEY_DUPLEX, 1.0, status_color, 2)
        
        # Telemetry
        conf_text = f"Confidence: {confidence*100:.1f}% | FPS: {fps:.1f}"
        cv2.putText(display, conf_text, (w - 380, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

        # Inset HOG gradient map in corner
        hog_vis_uint8 = exposure.rescale_intensity(hog_vis, in_range=(0, 10), out_range=(0, 255)).astype(np.uint8)
        hog_color = cv2.applyColorMap(hog_vis_uint8, cv2.COLORMAP_MAGMA)
        hog_thumb = cv2.resize(hog_color, (140, 140))
        
        # Draw inset border
        display[h - 155:h - 15, w - 155:w - 15] = hog_thumb
        cv2.rectangle(display, (w - 155, h - 155), (w - 15, h - 15), (255, 255, 255), 1)
        cv2.putText(display, "HOG Vector Field", (w - 155, h - 162), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow("Industrial Visual Inspection System - Bilal Butt", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if use_webcam:
        cap.release()
    cv2.destroyAllWindows()
    print("Real-time inspection terminated.")

if __name__ == '__main__':
    main()
