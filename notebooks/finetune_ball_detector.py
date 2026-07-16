"""Fine-tunes YOLOv8n's ball detection on SoccerSum, per Rosendo's decision
to prioritize broadcast-style framing (SoccerSum) over full-pitch panoramas
(SoccerTrack v2) -- see the internal task list "Future work / out of scope for now".

Player detection is already reasonable out of the box (F1=0.88 in the
earlier sanity check, notebooks/soccersum_layer1_validation.py); ball
recall was the weak point (F1=0.41, R=0.26). This fine-tunes on SoccerSum's
own Player/Goalkeeper/Referee/Ball boxes, remapped to our 2-class taxonomy
(0=person, 1=ball; Logo/Penalty Mark/Corner Flagpost/Goal Net dropped --
they're outside Layer 1's scope). Sequence 6114 is held out completely --
never touched during training -- so the before/after comparison against
notebooks/soccersum_layer1_validation.py's baseline numbers is fair.

Usage:
    python -m notebooks.finetune_ball_detector prepare   # build YOLO dataset
    python -m notebooks.finetune_ball_detector train     # fine-tune
    python -m notebooks.finetune_ball_detector eval       # evaluate vs held-out sequence 6114

Licensing note: SoccerSum is CC BY-NC-ND (No-Derivatives). The resulting
weights (WEIGHTS_OUT below) are a fine-tuned derivative of that data, so
they are gitignored and must stay local-only -- do not commit or
distribute them. src/perception/yolo_detector.py falls back to plain
pretrained COCO weights when this file is absent.
"""
from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ZIP_PATH = "data/raw/soccersum/Eliteserien.zip"
DATASET_DIR = Path("data/processed/soccersum_yolo")
HELD_OUT_SEQ = "6114"
WEIGHTS_OUT = Path("weights/soccersum_yolov8n_ball.pt")

# SoccerSum's own class ids (see simula/SoccerSum AnnotationProcessingTools/labelbox_to_yolo.py)
GT_PERSON_CLASSES = {0, 1, 2}  # Player, Goalkeeper, Referee
GT_BALL_CLASS = 3
# Every 6th non-held-out sequence goes to val, the rest to train.
VAL_EVERY = 6


def remap_label_line(line: str) -> str | None:
    parts = line.split()
    if not parts:
        return None
    cls = int(parts[0])
    if cls in GT_PERSON_CLASSES:
        new_cls = 0
    elif cls == GT_BALL_CLASS:
        new_cls = 1
    else:
        return None  # Logo / Penalty Mark / Corner Flagpost / Goal Net -- not our concern
    return " ".join([str(new_cls)] + parts[1:5])


def prepare() -> None:
    z = zipfile.ZipFile(ZIP_PATH)
    seq_ids = sorted({
        m.group(1) for n in z.namelist()
        if (m := re.match(r"Eliteserien/\d+/frames/(\d+)/", n))
    })
    seq_ids = [s for s in seq_ids if s != HELD_OUT_SEQ]
    val_ids = set(seq_ids[::VAL_EVERY])
    train_ids = set(seq_ids) - val_ids
    print(f"train sequences: {len(train_ids)}, val sequences: {len(val_ids)}, held out: {HELD_OUT_SEQ}")

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": [0, 0], "val": [0, 0]}  # [n_images, n_ball_labels]
    for n in z.namelist():
        m = re.match(r"Eliteserien/\d+/frames/(\d+)/(\d+_\d+)\.jpg", n)
        if not m:
            continue
        seq_id, stem = m.groups()
        if seq_id == HELD_OUT_SEQ:
            continue
        split = "val" if seq_id in val_ids else "train"

        img_bytes = z.read(n)
        (DATASET_DIR / "images" / split / f"{stem}.jpg").write_bytes(img_bytes)

        txt_name = n.replace("/frames/", "/detection/").rsplit(".", 1)[0] + ".txt"
        try:
            raw_lines = z.read(txt_name).decode().splitlines()
        except KeyError:
            raw_lines = []
        remapped = [r for line in raw_lines if (r := remap_label_line(line)) is not None]
        (DATASET_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(remapped))

        counts[split][0] += 1
        counts[split][1] += sum(1 for r in remapped if r.startswith("1 "))

    for split in ("train", "val"):
        n_img, n_ball = counts[split]
        print(f"{split}: {n_img} images, {n_ball} ball labels")

    yaml_path = DATASET_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {DATASET_DIR.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: person\n  1: ball\n"
    )
    print(f"Wrote {yaml_path}")


def train() -> None:
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.train(
        data=str((DATASET_DIR / "data.yaml").resolve()),
        epochs=30, imgsz=640, batch=16, device="cpu",
        project="data/processed/soccersum_yolo_runs", name="finetune",
        patience=10, verbose=True,
    )
    best = Path("data/processed/soccersum_yolo_runs/finetune/weights/best.pt")
    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, WEIGHTS_OUT)
    print(f"Copied fine-tuned weights to {WEIGHTS_OUT}")


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _match(preds: list, gts: list, iou_threshold: float) -> tuple[int, int, int, list]:
    unmatched_gt = list(range(len(gts)))
    tp, matched_ious = 0, []
    for p in preds:
        best_iou, best_idx = 0.0, None
        for gi in unmatched_gt:
            v = _iou(p, gts[gi])
            if v > best_iou:
                best_iou, best_idx = v, gi
        if best_idx is not None and best_iou >= iou_threshold:
            tp += 1
            matched_ious.append(best_iou)
            unmatched_gt.remove(best_idx)
    return tp, len(preds) - tp, len(unmatched_gt), matched_ious


def evaluate() -> None:
    """Same methodology as notebooks/soccersum_layer1_validation.py, but
    against the fine-tuned model, for a fair before/after comparison."""
    from ultralytics import YOLO
    from notebooks.soccersum_layer1_validation import load_gt_boxes

    model = YOLO(str(WEIGHTS_OUT))
    z = zipfile.ZipFile(ZIP_PATH)
    frame_names = sorted(
        n for n in z.namelist()
        if re.match(rf"Eliteserien/\d+/frames/{HELD_OUT_SEQ}/{HELD_OUT_SEQ}_\d+\.jpg", n)
    )

    extract_dir = Path(f"data/raw/soccersum/extracted/{HELD_OUT_SEQ}")
    extract_dir.mkdir(parents=True, exist_ok=True)

    thresholds = (0.3, 0.5)
    stats = {thr: {"person": [0, 0, 0], "ball": [0, 0, 0]} for thr in thresholds}
    all_ious = {"person": [], "ball": []}

    for img_name in frame_names:
        stem = img_name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        img_path = extract_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path.write_bytes(z.read(img_name))
        txt_path = extract_dir / f"{stem}.txt"
        if not txt_path.exists():
            det_name = img_name.replace("/frames/", "/detection/").rsplit(".", 1)[0] + ".txt"
            txt_path.write_bytes(z.read(det_name))

        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]
        gt_persons, gt_balls = load_gt_boxes(str(txt_path), w, h)

        result = model.predict(frame, verbose=False)[0]
        pred_persons, pred_balls = [], []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            xyxy = tuple(box.xyxy[0].tolist())
            (pred_persons if cls_id == 0 else pred_balls).append(xyxy)

        for thr in thresholds:
            tp, fp, fn, ious = _match(pred_persons, gt_persons, thr)
            s = stats[thr]["person"]; s[0] += tp; s[1] += fp; s[2] += fn
            if thr == thresholds[0]:
                all_ious["person"].extend(ious)
            tp, fp, fn, ious = _match(pred_balls, gt_balls, thr)
            s = stats[thr]["ball"]; s[0] += tp; s[1] += fp; s[2] += fn
            if thr == thresholds[0]:
                all_ious["ball"].extend(ious)

    for thr in thresholds:
        print(f"\n--- IoU threshold {thr} (fine-tuned model) ---")
        for cls in ("person", "ball"):
            tp, fp, fn = stats[thr][cls]
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            print(f"  {cls:8s}: TP={tp:4d} FP={fp:4d} FN={fn:4d}  P={precision:.2f} R={recall:.2f} F1={f1:.2f}")

    for cls in ("person", "ball"):
        vals = all_ious[cls]
        if vals:
            print(f"\nMean IoU of matched {cls} boxes (thr={thresholds[0]}): {np.mean(vals):.3f} (n={len(vals)})")
        else:
            print(f"\nNo matched {cls} boxes at thr={thresholds[0]}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    {"prepare": prepare, "train": train, "eval": evaluate}[cmd]()
