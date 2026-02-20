import cv2
import numpy as np
import math
import time
import argparse
from collections import deque
from typing import List, Tuple, Optional

class FPSMeter:
    def __init__(self, alpha: float = 0.2):
        self.last = None
        self.fps = None
        self.alpha = alpha

    def tick(self) -> Optional[float]:
        t = time.time()
        if self.last is None:
            self.last = t
            return None
        dt = t - self.last
        self.last = t
        if dt <= 0:
            return None
        inst = 1.0 / dt
        self.fps = inst if self.fps is None else (self.alpha * inst + (1 - self.alpha) * self.fps)
        return self.fps

def draw_text(img, text, xy=(12, 42), scale=1.0, color=(40, 220, 40), thickness=2):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# -----------------------------------------------------------------------------
# Single-file recognizer (segmentation, fingertip detection, classification)
# -----------------------------------------------------------------------------
class HandGestureRecognizer:
    def __init__(self,
                 hist_len: int = 12,
                 warmup_frames: int = 20,
                 min_area: int = 4000,
                 use_bg: bool = True):
        # Gesture labels
        self.gesture_names = {
            0: "Open Palm",
            1: "Fist",
            2: "Peace Sign",
            3: "Thumbs Up"
        }

        # History for temporal smoothing
        self.history = deque(maxlen=hist_len)

        # EMA for center stability
        self.center_ema = None
        self.alpha_center = 0.3

        # Parameters
        self.min_area = min_area
        self.use_bg = use_bg
        self.warmup_left = warmup_frames

        # Optional background subtractor (helps when background moves)
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=False) if use_bg else None

    # ---------- segmentation: fused skin detectors (YCrCb + HSV) ----------
    def _skin_mask_ycrcb(self, frame: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)
        Y = cv2.equalizeHist(Y)
        eq = cv2.merge([Y, Cr, Cb])
        lower = np.array([0, 135, 85], dtype=np.uint8)
        upper = np.array([255, 180, 135], dtype=np.uint8)
        return cv2.inRange(eq, lower, upper)

    def _skin_mask_hsv(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 30, 40], dtype=np.uint8)
        upper1 = np.array([25, 255, 255], dtype=np.uint8)
        lower2 = np.array([160, 30, 40], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)
        m1 = cv2.inRange(hsv, lower1, upper1)
        m2 = cv2.inRange(hsv, lower2, upper2)
        return cv2.bitwise_or(m1, m2)

    def _segment_hand(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # Slight blur to reduce noise
        blur = cv2.GaussianBlur(frame, (5, 5), 0)

        mask_y = self._skin_mask_ycrcb(blur)
        mask_h = self._skin_mask_hsv(blur)

        fused = cv2.bitwise_and(mask_y, mask_h)

        if self.use_bg and self.bg is not None:
            fg = self.bg.apply(blur)
            if self.warmup_left > 0:
                # Warmup: don't combine until background model stabilizes
                self.warmup_left -= 1
            else:
                fused = cv2.bitwise_and(fused, fg)

        # Morphology cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, kernel, iterations=1)
        fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, kernel, iterations=2)
        fused = cv2.dilate(fused, kernel, iterations=1)

        # Find largest contour (assume hand)
        cnts, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, fused

        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_area:
            return None, fused

        # Basic quality gating: solidity and extent
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) if hull is not None and len(hull) > 0 else 1.0
        solidity = area / (hull_area + 1e-6)
        x, y, w, h = cv2.boundingRect(c)
        extent = area / (w * h + 1e-6)
        if solidity < 0.7 or extent < 0.25:
            return None, fused

        return c, fused

    # ---------- fingertip detection using convexity defects & deduplication ----------
    def _detect_fingertips(self, contour: np.ndarray) -> Tuple[List[Tuple[int,int]], Tuple[int,int], List[Tuple]]:
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        if hull_idx is None or len(hull_idx) < 3:
            return [], (0,0), []

        defects = cv2.convexityDefects(contour, hull_idx)

        M = cv2.moments(contour)
        if M["m00"] == 0:
            return [], (0,0), []

        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        center = (cx, cy)

        # EMA for center to stabilize jitter
        if self.center_ema is None:
            self.center_ema = np.array(center, dtype=np.float32)
        else:
            self.center_ema = self.alpha_center * np.array(center, dtype=np.float32) + (1 - self.alpha_center) * self.center_ema
        center = (int(self.center_ema[0]), int(self.center_ema[1]))

        fingertip_candidates = []
        kept_defects = []

        if defects is not None:
            _, _, w, h = cv2.boundingRect(contour)
            hand_size = max(w, h)
            depth_thresh = max(9000, 0.03 * (hand_size ** 2))
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                a = np.linalg.norm(np.array(end) - np.array(start))
                b = np.linalg.norm(np.array(far) - np.array(start))
                c = np.linalg.norm(np.array(end) - np.array(far))
                if b * c == 0:
                    continue
                # Angle at far point
                angle = math.degrees(math.acos(np.clip((b**2 + c**2 - a**2) / (2*b*c), -1.0, 1.0)))
                if angle < 95 and d > depth_thresh:
                    kept_defects.append((start, end, far, d))
                    fingertip_candidates.extend([start, end])

        # Deduplicate nearby points
        dedup = []
        for p in fingertip_candidates:
            if all(math.hypot(p[0]-q[0], p[1]-q[1]) > 25 for q in dedup):
                dedup.append(p)

        # Keep points above (or near) palm center (common camera view)
        dedup = [p for p in dedup if p[1] < center[1] + 10]

        dedup.sort(key=lambda pt: pt[1])  # topmost first
        dedup = dedup[:5]  # max 5 fingers

        return dedup, center, kept_defects

    # ---------- classification rules ----------
    def _angle_between(self, p1: Tuple[int,int], p2: Tuple[int,int], center: Tuple[int,int]) -> float:
        v1 = np.array([p1[0]-center[0], p1[1]-center[1]], dtype=np.float32)
        v2 = np.array([p2[0]-center[0], p2[1]-center[1]], dtype=np.float32)
        n1 = np.linalg.norm(v1) + 1e-6
        n2 = np.linalg.norm(v2) + 1e-6
        cosang = np.clip(np.dot(v1, v2) / (n1*n2), -1.0, 1.0)
        return math.degrees(math.acos(cosang))

    def classify(self, fingertips: List[Tuple[int,int]], center: Tuple[int,int], contour: np.ndarray) -> int:
        num = len(fingertips)
        if contour is None:
            return -1

        x, y, w, h = cv2.boundingRect(contour)
        hand_sz = max(w, h)

        # 1) Open Palm: many fingertips and good spread
        if num >= 4:
            angles = []
            for i in range(num):
                for j in range(i+1, num):
                    angles.append(self._angle_between(fingertips[i], fingertips[j], center))
            if angles and np.percentile(angles, 75) > 35:
                return 0  # Open Palm

        # 2) Fist: no fingertips detected and compact contour
        if num == 0:
            area = cv2.contourArea(contour) + 1e-6
            per = cv2.arcLength(contour, True)
            circularity = 4 * math.pi * area / (per * per + 1e-6)
            if circularity > 0.5:
                return 1  # Fist

        # 3) Peace/V-sign: exactly 2 fingertips with sufficient separation & angle
        if num == 2:
            sep = math.hypot(fingertips[0][0]-fingertips[1][0], fingertips[0][1]-fingertips[1][1])
            ang = self._angle_between(fingertips[0], fingertips[1], center)
            if sep > 0.35 * hand_sz and 15 <= ang <= 60:
                if fingertips[0][1] < center[1] and fingertips[1][1] < center[1]:
                    return 2  # Peace

        # 4) Thumbs Up: single fingertip prominent above center with vertical orientation
        if num == 1:
            f = fingertips[0]
            dy = center[1] - f[1]   # positive when finger is above center
            dx = abs(f[0] - center[0])
            if dy > 0.25 * hand_sz and dx < 0.45 * hand_sz:
                # PCA-based major axis orientation
                data = contour.reshape(-1, 2).astype(np.float32)
                mean, eigvecs = cv2.PCACompute(data, mean=np.array([]))
                if eigvecs is not None and eigvecs.shape[0] > 0:
                    major = eigvecs[0]
                    # If major axis has a reasonable vertical component
                    if abs(major[1]) > 0.5:
                        return 3  # Thumbs Up

        return -1

    # ---------- full pipeline on a single frame ----------
    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, int, Optional[np.ndarray]]:
        contour, mask = self._segment_hand(frame_bgr)
        fingertips, center, defects = [], (0,0), []

        gesture = -1
        if contour is not None:
            # draw contour
            cv2.drawContours(frame_bgr, [contour], -1, (255, 0, 0), 2)

            # detect fingertips
            fingertips, center, defects = self._detect_fingertips(contour)
            for p in fingertips:
                cv2.circle(frame_bgr, p, 7, (0, 0, 255), -1)
            cv2.circle(frame_bgr, center, 6, (0, 255, 255), -1)

            # classify
            gesture = self.classify(fingertips, center, contour)
            self.history.append(gesture)

            # smooth by majority vote in history (ignore -1)
            votes = [g for g in self.history if g != -1]
            if votes:
                gesture = max(set(votes), key=votes.count)

        # overlay info
        if gesture != -1:
            draw_text(frame_bgr, self.gesture_names[gesture], xy=(12,42), scale=1.0, color=(40,220,40), thickness=2)
        draw_text(frame_bgr, f"Fingers: {len(fingertips)}", xy=(12,80), scale=0.7, color=(200,200,0), thickness=2)

        return frame_bgr, gesture, mask
        

# Main: argument parsing and capture loop


def parse_args():
    p = argparse.ArgumentParser(description="Hand Gesture Recognition (single-file)")
    p.add_argument("--camera", type=int, default=0, help="Webcam device index")
    p.add_argument("--no-bg", action="store_true", help="Disable background suppression (MOG2)")
    p.add_argument("--min-area", type=int, default=4000, help="Minimum hand area in pixels")
    p.add_argument("--warmup", type=int, default=20, help="Background subtractor warmup frames")
    p.add_argument("--width", type=int, default=960, help="Capture width")
    p.add_argument("--height", type=int, default=540, help="Capture height")
    p.add_argument("--show-mask", action="store_true", help="Show segmentation mask")
    return p.parse_args()

def main():
    args = parse_args()
    recognizer = HandGestureRecognizer(
        hist_len=12,
        warmup_frames=args.warmup,
        min_area=args.min_area,
        use_bg=not args.no_bg
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    print("Press 'q' to quit. Make sure your hand contrasts with the background.")
    fpsm = FPSMeter()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        fps = fpsm.tick()

        processed, gesture, mask = recognizer.process(frame)

        # draw FPS if available
        if fps is not None:
            cv2.putText(processed, f"{fps:.1f} FPS", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Hand Gesture Recognition", processed)
        if args.show_mask and mask is not None:
            cv2.imshow("Hand Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
