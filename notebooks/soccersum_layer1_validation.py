"""Layer 1 detection-quality sanity check against SoccerSum ground truth.

SoccerSum (Simula, Zenodo record 10612084, CC BY-NC-ND 4.0 -- non-commercial,
no redistribution of derivatives) provides 750 broadcast frames across 41
short bursts of 4-40 *contiguous* frames each (confirmed by inspecting frame
indices directly). The longest burst is 40 frames (~1-2s at broadcast fps),
so per the project's real-footage validation plan this dataset is used only
to sanity-check Layer 1's pretrained-YOLO detector against real broadcast
footage -- it is NOT used for Layers 2-4 or Module A, which need continuous
multi-second tracking that no SoccerSum sequence is long enough to provide.

Ground truth is YOLO-format boxes for 8 classes (Player, Goalkeeper,
Referee, Ball, Logo, Penalty Mark, Corner Flagpost, Goal Net -- mapping
confirmed from simula/SoccerSum's AnnotationProcessingTools/labelbox_to_yolo.py).
Our detector only distinguishes COCO "person" vs "sports ball", so
Player+Goalkeeper+Referee are merged into one "person" ground-truth class
for comparison; Logo/Penalty Mark/Corner Flagpost/Goal Net are outside
Layer 1's scope and ignored.
"""
from __future__ import annotations

import glob
import os

import cv2
import numpy as np

from src.perception.yolo_detector import detect_frame

SEQ_DIR = "data/raw/soccersum/extracted/6114"
GT_PERSON_CLASSES = {0, 1, 2}  # Player, Goalkeeper, Referee
GT_BALL_CLASS = 3
IOU_THRESHOLDS = (0.3, 0.5)


def load_gt_boxes(txt_path: str, img_w: int, img_h: int) -> tuple[list, list]:
    persons, balls = [], []
    if not os.path.exists(txt_path):
        return persons, balls
    with open(txt_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            cls = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:5])
            box = (
                (cx - w / 2) * img_w, (cy - h / 2) * img_h,
                (cx + w / 2) * img_w, (cy + h / 2) * img_h,
            )
            if cls in GT_PERSON_CLASSES:
                persons.append(box)
            elif cls == GT_BALL_CLASS:
                balls.append(box)
    return persons, balls


def iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def match(preds: list, gts: list, iou_threshold: float) -> tuple[int, int, int, list]:
    """Greedy IoU matching. Returns (tp, fp, fn, matched_ious)."""
    unmatched_gt = list(range(len(gts)))
    tp, matched_ious = 0, []
    for p in preds:
        best_iou, best_idx = 0.0, None
        for gi in unmatched_gt:
            v = iou(p, gts[gi])
            if v > best_iou:
                best_iou, best_idx = v, gi
        if best_idx is not None and best_iou >= iou_threshold:
            tp += 1
            matched_ious.append(best_iou)
            unmatched_gt.remove(best_idx)
    return tp, len(preds) - tp, len(unmatched_gt), matched_ious


def run() -> None:
    frame_paths = sorted(glob.glob(f"{SEQ_DIR}/*.jpg"))
    print(f"Evaluating {len(frame_paths)} frames from sequence {os.path.basename(SEQ_DIR)}")

    stats = {thr: {"person": [0, 0, 0], "ball": [0, 0, 0]} for thr in IOU_THRESHOLDS}
    all_ious = {"person": [], "ball": []}
    per_frame_report = []

    for i, img_path in enumerate(frame_paths):
        frame = cv2.imread(img_path)
        h, w = frame.shape[:2]
        txt_path = img_path.rsplit(".", 1)[0] + ".txt"
        gt_persons, gt_balls = load_gt_boxes(txt_path, w, h)

        raw_boxes = detect_frame(frame, i)
        pred_persons = [(b.x1, b.y1, b.x2, b.y2) for b in raw_boxes if b.cls == "person"]
        pred_balls = [(b.x1, b.y1, b.x2, b.y2) for b in raw_boxes if b.cls == "ball"]

        frame_row = {"frame": os.path.basename(img_path), "n_gt_person": len(gt_persons),
                     "n_pred_person": len(pred_persons), "n_gt_ball": len(gt_balls),
                     "n_pred_ball": len(pred_balls)}
        per_frame_report.append(frame_row)

        for thr in IOU_THRESHOLDS:
            tp, fp, fn, ious = match(pred_persons, gt_persons, thr)
            s = stats[thr]["person"]
            s[0] += tp; s[1] += fp; s[2] += fn
            if thr == IOU_THRESHOLDS[0]:
                all_ious["person"].extend(ious)

            tp, fp, fn, ious = match(pred_balls, gt_balls, thr)
            s = stats[thr]["ball"]
            s[0] += tp; s[1] += fp; s[2] += fn
            if thr == IOU_THRESHOLDS[0]:
                all_ious["ball"].extend(ious)

    print("\nPer-frame detection counts (first 10 frames):")
    for row in per_frame_report[:10]:
        print(" ", row)

    for thr in IOU_THRESHOLDS:
        print(f"\n--- IoU threshold {thr} ---")
        for cls in ("person", "ball"):
            tp, fp, fn = stats[thr][cls]
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            print(f"  {cls:8s}: TP={tp:4d} FP={fp:4d} FN={fn:4d}  P={precision:.2f} R={recall:.2f} F1={f1:.2f}")

    for cls in ("person", "ball"):
        vals = all_ious[cls]
        if vals:
            print(f"\nMean IoU of matched {cls} boxes (thr={IOU_THRESHOLDS[0]}): {np.mean(vals):.3f} (n={len(vals)})")
        else:
            print(f"\nNo matched {cls} boxes at thr={IOU_THRESHOLDS[0]}")


if __name__ == "__main__":
    run()
