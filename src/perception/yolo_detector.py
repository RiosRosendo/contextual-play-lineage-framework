"""YOLOv8 detection path for real broadcast footage. Prefers a SoccerSum
fine-tuned checkpoint (weights/soccersum_yolov8n_ball.pt, 2-class:
person/ball) when present -- it noticeably improves ball recall over the
plain COCO-pretrained model (F1 0.41 -> 0.60 on a held-out SoccerSum
sequence; see notebooks/finetune_ball_detector.py).

That weight file is optional and local-only -- it is NOT committed to this
repo and NOT distributed anywhere. SoccerSum is licensed CC BY-NC-ND
(No-Derivatives), and a model fine-tuned on it is arguably a derivative
work, so shipping the resulting weights publicly would be a real licensing
risk. If you want the fine-tuned detector, download SoccerSum yourself
(see notebooks/finetune_ball_detector.py's docstring for the Zenodo
record) and run `python -m notebooks.finetune_ball_detector prepare` then
`train` to regenerate it locally -- it's gitignored by `*.pt` like any
other weight file. Without it, this module falls back cleanly to plain
pretrained COCO classes (person=0, sports ball=32), so everything still
works on a fresh clone with no extra setup.

Neither model distinguishes players from the referee yet -- that needs
further fine-tuning (tracked in TODO.md). Team split is delegated to
team_id.py on top of the person boxes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_COCO_PERSON = 0
_COCO_SPORTS_BALL = 32
_FINE_TUNED_WEIGHTS = Path("weights/soccersum_yolov8n_ball.pt")

_model = None
_using_fine_tuned = False


def _get_model():
    global _model, _using_fine_tuned
    if _model is None:
        from ultralytics import YOLO
        if _FINE_TUNED_WEIGHTS.exists():
            _model = YOLO(str(_FINE_TUNED_WEIGHTS))
            _using_fine_tuned = True
        else:
            _model = YOLO("yolov8n.pt")
            _using_fine_tuned = False
    return _model


@dataclass
class RawBox:
    frame_idx: int
    cls: str  # "person" | "ball"
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float


def available() -> bool:
    try:
        _get_model()
        return True
    except Exception:
        return False


def detect_frame(frame_bgr: np.ndarray, frame_idx: int) -> list[RawBox]:
    model = _get_model()
    if _using_fine_tuned:
        results = model.predict(frame_bgr, verbose=False)[0]
        boxes: list[RawBox] = []
        for box in results.boxes:
            cls = "person" if int(box.cls[0]) == 0 else "ball"
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append(RawBox(frame_idx, cls, x1, y1, x2, y2, float(box.conf[0])))
        return boxes

    results = model.predict(
        frame_bgr, classes=[_COCO_PERSON, _COCO_SPORTS_BALL], verbose=False
    )[0]
    boxes = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls = "person" if cls_id == _COCO_PERSON else "ball"
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append(RawBox(frame_idx, cls, x1, y1, x2, y2, float(box.conf[0])))
    return boxes
