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
  - Real-time confidence gauge, inference latency & inspection FPS
  - Inset HOG gradient vector visualization
"""

import cv2
import time
import glob
import os
import argparse
import numpy as np
from skimage.feature import hog
from skimage import exposure
from sklearn.svm import SVC
import imageio

def train_baseline_model():
    data_dir = "data"
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    if not image_paths:
        print("[!] Data directory empty or missing.")
        return None, []

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
    print(f"[+] Baseline SVM model calibrated from {len(X)} industrial samples.")
    return clf, image_paths

def render_hud_frame(display_bgr, cat_name, is_defective, confidence, fps, latency_ms, hog_vis):
    h, w, _ = display_bgr.shape
    canvas = display_bgr.copy()
    overlay = canvas.copy()

    status_color = (40, 40, 230) if is_defective else (50, 205, 50)  # BGR
    status_text = "STATUS: REJECT [DEFECTIVE]" if is_defective else "STATUS: ACCEPT [PASS]"
    decision_badge = "REJECT" if is_defective else "ACCEPT"

    # Semi-transparent top HUD bar
    cv2.rectangle(overlay, (0, 0), (w, 75), (20, 24, 33), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

    # Accent top border
    cv2.rectangle(canvas, (0, 0), (w, 75), status_color, 2)
    cv2.putText(canvas, status_text, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 0.95, status_color, 2)
    
    # Telemetry metrics
    cv2.putText(canvas, f"INSPECTION: {cat_name.upper()}", (22, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    conf_text = f"Confidence: {confidence*100:.1f}% | Latency: {latency_ms:.1f}ms | FPS: {fps:.1f}"
    cv2.putText(canvas, conf_text, (w - 440, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)

    # Bottom industrial telemetry bar
    overlay2 = canvas.copy()
    cv2.rectangle(overlay2, (0, h - 35), (w, h), (15, 18, 25), -1)
    cv2.addWeighted(overlay2, 0.85, canvas, 0.15, 0, canvas)
    cv2.putText(canvas, "AOI LINE-01 | HOG 8x8 cell (9 bins) | Linear SVM | QC GATE", (20, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 180, 200), 1)

    # Inset HOG gradient vector field
    hog_vis_uint8 = exposure.rescale_intensity(hog_vis, in_range=(0, 10), out_range=(0, 255)).astype(np.uint8)
    hog_color = cv2.applyColorMap(hog_vis_uint8, cv2.COLORMAP_MAGMA)
    hog_thumb = cv2.resize(hog_color, (140, 140))
    
    # Draw inset border
    canvas[h - 185:h - 45, w - 160:w - 20] = hog_thumb
    cv2.rectangle(canvas, (w - 160, h - 185), (w - 20, h - 45), (255, 255, 255), 2)
    cv2.putText(canvas, "HOG VECTOR MAP", (w - 160, h - 192), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    return canvas

def main():
    parser = argparse.ArgumentParser(description="Real-Time Industrial Defect Inspector HUD")
    parser.add_argument("--demo-frames", type=int, default=0, help="Record N frames to animated GIF")
    parser.add_argument("--save-gif", type=str, default="figures/realtime_hud_demo.gif", help="GIF output path")
    parser.add_argument("--no-gui", action="store_true", help="Run without opening GUI window")
    args = parser.parse_args()

    clf, image_paths = train_baseline_model()
    if clf is None:
        return

    # Attempt to open hardware webcam if not running headless demo
    use_webcam = False
    cap = None
    if not args.no_gui and args.demo_frames == 0:
        cap = cv2.VideoCapture(0)
        use_webcam = cap.isOpened()
    
    # Prepare simulated surface stream
    # Ensure alternating mix of normal and defective samples for rich demonstration
    normals = [p for p in image_paths if 'normal' in os.path.basename(p)]
    defects = [p for p in image_paths if 'normal' not in os.path.basename(p)]
    
    stream_samples = []
    max_len = max(len(normals), len(defects))
    for i in range(max_len):
        if defects:
            stream_samples.append(defects[i % len(defects)])
        if normals:
            stream_samples.append(normals[i % len(normals)])

    print("\n=======================================================")
    print("  AUTOMATED OPTICAL INSPECTION (AOI) - REAL-TIME HUD   ")
    print(f"  Mode: {'HARDWARE CAMERA' if use_webcam else 'SIMULATED CONVEYOR STREAM'}")
    if args.demo_frames > 0:
        print(f"  Recording {args.demo_frames} demo frames to {args.save_gif}")
    else:
        print("  Press 'q' to stop inspection prototype.")
    print("=======================================================\n")

    frame_count = 0
    sim_idx = 0
    fps = 30.0
    recorded_frames = []

    total_frames_target = args.demo_frames if args.demo_frames > 0 else 9999999

    while frame_count < total_frames_target:
        loop_start = time.perf_counter()
        if use_webcam:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crop = cv2.resize(gray, (128, 128))
            display = frame.copy()
            cat_name = "live_camera"
        else:
            sim_path = stream_samples[sim_idx % len(stream_samples)]
            sim_idx += 1
            cat_name = os.path.basename(sim_path).rsplit('_', 1)[0]
            raw = cv2.imread(sim_path)
            display = cv2.resize(raw, (640, 480))
            gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
            crop = cv2.resize(gray, (128, 128))

        # HOG Extraction & Visualization
        t_hog = time.perf_counter()
        feat, hog_vis = hog(crop, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                            block_norm='L2-Hys', visualize=True)
        feat = feat.reshape(1, -1)

        # Classification
        probs = clf.predict_proba(feat)[0]
        prob_defective = probs[1]
        is_defective = prob_defective >= 0.50
        confidence = prob_defective if is_defective else probs[0]
        latency_ms = (time.perf_counter() - t_hog) * 1000.0

        # Construct Frame
        canvas = render_hud_frame(display, cat_name, is_defective, confidence, fps, latency_ms, hog_vis)
        frame_count += 1

        elapsed = time.perf_counter() - loop_start
        fps = 1.0 / max(elapsed, 0.001)

        if args.demo_frames > 0:
            # Convert BGR to RGB for GIF recording
            rgb_frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            recorded_frames.append(rgb_frame)
            if frame_count % 5 == 0:
                print(f"  [*] Processed {frame_count}/{args.demo_frames} frames: {cat_name} -> {'REJECT' if is_defective else 'ACCEPT'} ({confidence*100:.1f}%)")
        else:
            if not args.no_gui:
                cv2.imshow("Industrial Visual Inspection System - Bilal Butt", canvas)
                key = cv2.waitKey(120) & 0xFF
                if key == ord('q'):
                    break

    if use_webcam and cap is not None:
        cap.release()
    if not args.no_gui:
        cv2.destroyAllWindows()

    if args.demo_frames > 0 and recorded_frames:
        os.makedirs(os.path.dirname(args.save_gif), exist_ok=True)
        print(f"\n[+] Saving animated inspection GIF to {args.save_gif}...")
        imageio.mimsave(args.save_gif, recorded_frames, fps=4, loop=0)
        
        # Also save high-res static snapshot
        snapshot_path = os.path.join(os.path.dirname(args.save_gif), "realtime_hud_snapshot.png")
        cv2.imwrite(snapshot_path, canvas)
        print(f"[+] Saved snapshot to {snapshot_path}")

    print("[+] Real-time inspection prototype complete.")

if __name__ == '__main__':
    main()
