"""
================================================================================
LAB 05 BONUS CHALLENGE: High-Definition Industrial Defect Inspection HUD
Author: Bilal Butt (@TheBilalButt)
Course: Computer Vision Lab 05
================================================================================

Features:
  - Dual-View High-Definition Interface (Main Camera Stream + Zoomed ROI + HOG Map)
  - Interactive Central Inspection Reticle (Corner brackets with targeting crosshair)
  - Real-Time Adaptive Contrast Enhancement (CLAHE) for low-light webcam rooms
  - Live Stream Modes: Switch between LIVE WEBCAM and SIMULATED CONVEYOR FEED
  - Production Quality Control (QC) Telemetry: PASS / REJECT Gating, Latency, FPS
  - Fully Resizable Window (cv2.WINDOW_NORMAL) - maximize to fullscreen anytime!

Keyboard Controls:
  [m]      : Toggle between LIVE WEBCAM and SIMULATED CONVEYOR FEED
  [b]      : Toggle Display Contrast Boost (ON / OFF)
  [+] / [-]: Increase / Decrease Inspection Reticle Box Size
  [t]      : Cycle Defect Sensitivity Threshold (30%, 50%, 70%)
  [s]      : Save inspection snapshot to figures/
  [SPACE]  : Freeze / Resume inspection frame
  [q]      : Exit inspection system
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

def train_calibrated_model():
    """Trains a calibrated Linear SVM model directly on standardized industrial HOG features."""
    data_dir = "data"
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    if not image_paths:
        print("[!] Probing alternate directory...")
        image_paths = sorted(glob.glob(os.path.join("..", "data", "*.jpg")))

    if not image_paths:
        print("[ERROR] No dataset images found for calibration!")
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
    print(f"[+] Linear SVM model calibrated successfully on {len(X)} industrial samples.")
    return clf, image_paths

def draw_corner_brackets(img, pt1, pt2, color, thickness=2, length=24):
    """Draws sleek targeting brackets around the inspection region."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

def main():
    parser = argparse.ArgumentParser(description="High-Definition Industrial Defect Inspection HUD")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--demo-frames", type=int, default=0, help="Record N frames to animated GIF")
    parser.add_argument("--save-gif", type=str, default="figures/realtime_hud_demo.gif", help="GIF output path")
    parser.add_argument("--no-gui", action="store_true", help="Headless execution without GUI window")
    parser.add_argument("--force-conveyor", action="store_true", help="Start directly in simulated conveyor mode")
    args = parser.parse_args()

    clf, image_paths = train_calibrated_model()
    if clf is None:
        return

    # Attempt camera initialization
    cap = None
    use_webcam = False
    if not args.no_gui and args.demo_frames == 0 and not args.force_conveyor:
        print(f"[*] Probing camera index {args.camera}...")
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(args.camera)
        use_webcam = cap.isOpened()
        if use_webcam:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"[+] Webcam {args.camera} online (640x480 resolution).")
        else:
            print("[!] Webcam not available. Defaulting to SIMULATED CONVEYOR STREAM.")

    # Prepare alternating conveyor stream
    normals = [p for p in image_paths if 'normal' in os.path.basename(p)]
    defects = [p for p in image_paths if 'normal' not in os.path.basename(p)]
    sim_samples = []
    max_len = max(len(normals), len(defects))
    for i in range(max_len):
        if defects:
            sim_samples.append(defects[i % len(defects)])
        if normals:
            sim_samples.append(normals[i % len(normals)])

    # Inspection settings
    roi_size = 220
    clahe_boost = True
    clahe_obj = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    threshold_idx = 1
    thresholds = [0.30, 0.50, 0.70]
    paused = False
    last_frame = None

    window_name = "Industrial Surface Defect Inspection HUD - Bilal Butt"
    if not args.no_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1080, 680)

    print("\n" + "=" * 72)
    print("   AUTOMATED OPTICAL INSPECTION (AOI) - HIGH-DEFINITION LIVE HUD   ")
    print(f"   Active Stream: {'LIVE HARDWARE WEBCAM' if use_webcam else 'SIMULATED CONVEYOR STREAM'}")
    print("   Interactive Hotkeys:")
    print("     [m] Switch Mode (Webcam <-> Conveyor) | [b] Toggle Contrast Boost")
    print("     [+] / [-] Resize Reticle Box          | [s] Save Inspection Snapshot")
    print("     [SPACE] Freeze / Resume Inspection    | [q] Exit Inspection System")
    print("=" * 72 + "\n")

    frame_count = 0
    sim_idx = 0
    current_fps = 30.0
    recorded_frames = []
    max_demo_frames = args.demo_frames if args.demo_frames > 0 else 9999999

    while frame_count < max_demo_frames:
        loop_start = time.perf_counter()

        # 1. Image Acquisition
        if not paused or last_frame is None:
            if use_webcam and cap is not None and cap.isOpened():
                ret, raw_bgr = cap.read()
                if not ret:
                    use_webcam = False
                    raw_bgr = cv2.imread(sim_samples[0])
                else:
                    raw_bgr = cv2.resize(raw_bgr, (640, 480))
                source_tag = "LIVE WEBCAM"
                cat_label = "live_sample"
                h, w, _ = raw_bgr.shape
                cx, cy = w // 2, h // 2
                half_box = roi_size // 2
                x1 = max(0, cx - half_box)
                y1 = max(0, cy - half_box)
                x2 = min(w, cx + half_box)
                y2 = min(h, cy + half_box)
                roi_gray = cv2.cvtColor(raw_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            else:
                sim_path = sim_samples[sim_idx % len(sim_samples)]
                sim_idx += 1
                cat_label = os.path.basename(sim_path).rsplit('_', 1)[0]
                plate_img = cv2.imread(sim_path)
                
                # Render simulated conveyor background
                raw_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
                raw_bgr[:, :] = (30, 35, 42)
                for y_line in range(0, 480, 40):
                    cv2.line(raw_bgr, (0, y_line), (640, y_line), (42, 48, 58), 1)
                
                # Place plate nicely into conveyor view
                plate_display = cv2.resize(plate_img, (roi_size, roi_size))
                h, w, _ = raw_bgr.shape
                cx, cy = w // 2, h // 2
                half_box = roi_size // 2
                x1 = cx - half_box
                y1 = cy - half_box
                x2 = cx + half_box
                y2 = cy + half_box
                raw_bgr[y1:y2, x1:x2] = plate_display
                
                source_tag = f"CONVEYOR: {cat_label.upper()}"
                roi_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            
            last_frame = (raw_bgr.copy(), roi_gray.copy(), source_tag, cat_label, x1, y1, x2, y2)
        else:
            raw_bgr, roi_gray, source_tag, cat_label, x1, y1, x2, y2 = last_frame

        h, w, _ = raw_bgr.shape
        cx, cy = w // 2, h // 2

        # 2. HOG Feature Extraction on standardized grayscale ROI
        t_hog = time.perf_counter()
        roi_std = cv2.resize(roi_gray, (128, 128))
        feat, hog_vis = hog(roi_std, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        feat = feat.reshape(1, -1)

        # 3. Classification & QC Logic
        probs = clf.predict_proba(feat)[0]
        prob_defect = probs[1]
        active_thresh = thresholds[threshold_idx]
        is_defective = prob_defect >= active_thresh
        confidence = prob_defect if is_defective else probs[0]
        latency_ms = (time.perf_counter() - t_hog) * 1000.0

        # Status Colors
        status_color = (40, 45, 235) if is_defective else (50, 210, 60)  # Red / Green
        status_text = "STATUS: REJECT [DEFECTIVE]" if is_defective else "STATUS: ACCEPT [DEFECT-FREE]"

        # 4. Build Wide-Screen Industrial Canvas (1040 x 620)
        canvas = np.zeros((620, 1040, 3), dtype=np.uint8)
        canvas[:, :] = (20, 24, 30)  # Dark metallic background

        # Left View: Main Stream
        cam_display = raw_bgr.copy()
        cv2.rectangle(cam_display, (x1, y1), (x2, y2), status_color, 2)
        draw_corner_brackets(cam_display, (x1, y1), (x2, y2), (255, 255, 255), thickness=3, length=24)
        cv2.drawMarker(cam_display, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1)
        cv2.putText(cam_display, "[ PLACE SURFACE IN RETICLE ]", (cx - 130, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        canvas[80:560, 20:660] = cam_display

        # Right Panel: Zoomed Inspection ROI Preview (with CLAHE for display clarity)
        if clahe_boost:
            roi_display = clahe_obj.apply(roi_gray)
        else:
            roi_display = roi_gray
        roi_preview = cv2.resize(cv2.cvtColor(roi_display, cv2.COLOR_GRAY2BGR), (340, 220))
        cv2.rectangle(roi_preview, (0, 0), (339, 219), (70, 80, 95), 2)
        cv2.putText(roi_preview, f"ZOOMED ROI ({'CLAHE BOOST' if clahe_boost else 'RAW'})", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
        canvas[80:300, 680:1020] = roi_preview

        # Right Panel: HOG Gradient Energy Map
        hog_vis_u8 = exposure.rescale_intensity(hog_vis, in_range=(0, 10), out_range=(0, 255)).astype(np.uint8)
        hog_color = cv2.applyColorMap(hog_vis_u8, cv2.COLORMAP_MAGMA)
        hog_preview = cv2.resize(hog_color, (340, 240))
        cv2.rectangle(hog_preview, (0, 0), (339, 239), (70, 80, 95), 2)
        cv2.putText(hog_preview, "HOG GRADIENT VECTOR MAP", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)
        canvas[320:560, 680:1020] = hog_preview

        # Top Industrial Header Bar
        cv2.rectangle(canvas, (0, 0), (1040, 65), (28, 32, 42), -1)
        cv2.line(canvas, (0, 65), (1040, 65), status_color, 3)
        cv2.putText(canvas, status_text, (25, 42), cv2.FONT_HERSHEY_DUPLEX, 0.95, status_color, 2)
        telemetry_str = f"CONF: {confidence*100:.1f}%  |  LATENCY: {latency_ms:.1f}ms  |  FPS: {current_fps:.1f}"
        cv2.putText(canvas, telemetry_str, (460, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2)

        # Bottom Information & Controls Bar
        cv2.rectangle(canvas, (0, 575), (1040, 620), (16, 20, 26), -1)
        info_line1 = f"STREAM: [{source_tag}]  |  CLAHE BOOST: [{'ON' if clahe_boost else 'OFF'}]  |  THRESH: {active_thresh*100:.0f}%  |  BOX: {roi_size}px"
        info_line2 = "HOTKEYS: [m] Toggle Webcam/Conveyor | [b] Boost | [+/-] Box Size | [s] Snapshot | [q] Exit"
        cv2.putText(canvas, info_line1, (25, 595), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 200, 220), 1)
        cv2.putText(canvas, info_line2, (25, 612), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 150, 170), 1)

        # Telemetry updates
        frame_count += 1
        elapsed = time.perf_counter() - loop_start
        current_fps = 1.0 / max(elapsed, 0.001)

        if args.demo_frames > 0:
            rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            recorded_frames.append(rgb_canvas)
            if frame_count % 5 == 0:
                print(f"  [*] Captured {frame_count}/{args.demo_frames} frames ({source_tag}) -> {status_text}")
        else:
            if not args.no_gui:
                cv2.imshow(window_name, canvas)
                key = cv2.waitKey(25 if use_webcam else 120) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"[*] Inspection {'PAUSED' if paused else 'RESUMED'}")
                elif key == ord('m'):
                    if use_webcam:
                        use_webcam = False
                        print("[*] Switched to SIMULATED CONVEYOR STREAM.")
                    else:
                        if cap is not None and cap.isOpened():
                            use_webcam = True
                            print("[*] Switched to LIVE WEBCAM.")
                        else:
                            cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
                            if not cap.isOpened():
                                cap = cv2.VideoCapture(args.camera)
                            use_webcam = cap.isOpened()
                            if use_webcam:
                                print("[*] Webcam opened successfully.")
                            else:
                                print("[!] Webcam unavailable.")
                elif key == ord('b'):
                    clahe_boost = not clahe_boost
                    print(f"[*] Contrast Boost (CLAHE): {'ENABLED' if clahe_boost else 'DISABLED'}")
                elif key in [ord('+'), ord('=')]:
                    roi_size = min(360, roi_size + 20)
                    print(f"[*] Inspection reticle box resized to {roi_size}px")
                elif key in [ord('-'), ord('_')]:
                    roi_size = max(120, roi_size - 20)
                    print(f"[*] Inspection reticle box resized to {roi_size}px")
                elif key == ord('t'):
                    threshold_idx = (threshold_idx + 1) % len(thresholds)
                    print(f"[*] Defect threshold set to {thresholds[threshold_idx]*100:.0f}%")
                elif key == ord('s'):
                    snap_name = f"figures/inspection_snapshot_{int(time.time())}.png"
                    cv2.imwrite(snap_name, canvas)
                    print(f"[+] Snapshot saved to: {snap_name}")

    if cap is not None and cap.isOpened():
        cap.release()
    if not args.no_gui:
        cv2.destroyAllWindows()

    if args.demo_frames > 0 and recorded_frames:
        os.makedirs(os.path.dirname(args.save_gif), exist_ok=True)
        print(f"\n[+] Writing animated demo GIF to {args.save_gif}...")
        imageio.mimsave(args.save_gif, recorded_frames, fps=4, loop=0)
        snap_path = os.path.join(os.path.dirname(args.save_gif), "realtime_hud_snapshot.png")
        cv2.imwrite(snap_path, canvas)
        print(f"[+] Saved high-resolution HUD snapshot to {snap_path}")

    print("[+] Inspection system exited cleanly.")

if __name__ == '__main__':
    main()
