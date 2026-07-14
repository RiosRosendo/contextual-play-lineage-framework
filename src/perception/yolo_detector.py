"""Pretrained-YOLOv8 detection path for real broadcast footage. Uses COCO
classes (person=0, sports ball=32) with no fine-tuning, per the "simplest
possible version first" philosophy -- it cannot yet distinguish players from
the referee, or tell teams apart (that needs SoccerNet fine-tuning, tracked
in TODO.md). Team split is delegated to team_id.py on top of the person boxes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_COCO_PERSON = 0
_COCO_SPORTS_BALL = 32

_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")
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
    results = model.predict(
        frame_bgr, classes=[_COCO_PERSON, _COCO_SPORTS_BALL], verbose=False
    )[0]
    boxes: list[RawBox] = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls = "person" if cls_id == _COCO_PERSON else "ball"
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append(RawBox(frame_idx, cls, x1, y1, x2, y2, float(box.conf[0])))
    return boxes
