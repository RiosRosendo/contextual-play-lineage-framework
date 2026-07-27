"""Second, identity-only detection signal (2026-07-20): a pretrained
multi-class detector (ball/goalkeeper/player/referee), trained on the
DFL-Bundesliga-Data-Shootout broadcast dataset (Roboflow's
"football-players-detection", CC BY 4.0 -- see PROGRESS.md's 2026-07-20
entries for the full evaluation this module implements the conclusion of).

Deliberately NOT used for box existence/recall. Real-clip testing found a
severe, reproducible recall gap on close-contact/zoomed-in frames (tackles,
tangles) across every clip tested (Swansea, Chelsea-Burnley, Leicester) --
root-caused to a shot-scale effect, not a clip-specific quirk: this model's
own training data evidently skews toward wide tactical framing, and it
reliably fails to find real, full-body players once the broadcast zooms in
close enough (confirmed via a clean, order-of-magnitude gap in box-size
statistics between frames where it worked and frames where it didn't).

Only used as a SECOND opinion for classifying boxes the project's own
primary detector (yolo_detector.py) already found, and only on frames
`pipeline.py`'s scale gate identifies as "wide" -- see
`PERSON_BOX_MEAN_AREA_FRAC_THRESHOLD` there. This module has no opinion on
whether a box exists at all, only on what an already-existing box is.

WEIGHTS_PATH points to the downloaded .pt file (gitignored, not committed --
same convention as yolo_detector.py's fine-tuned-weights handling and
kit_pattern_classifier.py/player_classifier.py's trained-head weights).
Returns None (graceful no-op, same convention as those modules) for every
box if the weights aren't present in this checkout."""
from __future__ import annotations

from pathlib import Path

WEIGHTS_PATH = Path("weights/roboflow_player_referee_detector.pt")

# Two independently-trained detectors won't box the same real person
# identically -- moderate threshold, matching pose_estimator.POSE_MATCH_IOU's
# own reasoning for the same kind of cross-model box association.
MATCH_IOU = 0.3

# Confidence floor for a Roboflow detection to even be considered as a match
# candidate -- kept low (this model's own confident detections on frames it
# handles well are typically >0.7; the real gate keeping bad frames out is
# pipeline.py's shot-scale check, not this threshold).
MIN_CONF = 0.1

_model = None
_checked = False


def _get_model():
    global _model, _checked
    if _checked:
        return _model
    _checked = True
    if not WEIGHTS_PATH.exists():
        return None
    from ultralytics import YOLO
    print("Loading Roboflow referee/goalkeeper detector (first call only)...")
    _model = YOLO(str(WEIGHTS_PATH))
    print("Loaded Roboflow referee/goalkeeper detector.")
    return _model


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def classify_boxes(frame_bgr, person_boxes: list[tuple]) -> list[str | None]:
    """Returns, per input box (same order/length as `person_boxes`, each an
    (x1,y1,x2,y2) tuple): "referee", "goalkeeper", "player", or None -- no
    confident Roboflow match at all, meaning the caller's existing
    color-based logic should decide on its own, exactly as if this signal
    didn't exist. Callers are expected to only invoke this on frames
    confirmed wide enough for the model to be reliable (see
    `pipeline.py`'s scale gate) -- this function itself has no opinion on
    shot scale, it just answers "what does this model see," to keep the
    scale-vs-classification concerns cleanly separated."""
    model = _get_model()
    if model is None or not person_boxes:
        return [None] * len(person_boxes)
    results = model(frame_bgr, verbose=False, conf=MIN_CONF)[0]
    rf_boxes = []
    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])]
        if cls_name == "ball":
            continue  # ball identity isn't this module's job -- see docstring
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        rf_boxes.append((cls_name, (x1, y1, x2, y2)))

    labels: list[str | None] = []
    for pb in person_boxes:
        best_iou, best_cls = 0.0, None
        for cls_name, rb in rf_boxes:
            i = _iou(pb, rb)
            if i > best_iou:
                best_iou, best_cls = i, cls_name
        labels.append(best_cls if best_iou >= MATCH_IOU else None)
    return labels
