"""Fallback detector used when a YOLO backend isn't available or performs
poorly on non-photorealistic footage (our synthetic test clip). HSV blob
detection over the known jersey/ball colors. Deliberately simple per
CLAUDE.md section 3 ("simplified formulas" for the first end-to-end pass).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# HSV ranges tuned to the synthetic clip's fixed BGR palette
# (src/perception/synthetic_clip.py). Real-footage jersey colors would need
# per-match calibration -- tracked in TODO.md as a Layer 1 deepening item.
_COLOR_RANGES = {
    "team_a": ((0, 120, 120), (10, 255, 255)),      # red
    "team_b": ((100, 120, 100), (120, 255, 255)),   # blue
    "referee": ((25, 120, 120), (35, 255, 255)),    # yellow
    "ball": ((11, 150, 150), (22, 255, 255)),        # orange (distinct from white pitch lines)
}

_CLASS_FOR_COLOR = {
    "team_a": "player",
    "team_b": "player",
    "referee": "referee",
    "ball": "ball",
}


@dataclass
class Detection:
    frame_idx: int
    cls: str  # "player" | "referee" | "ball"
    team_hint: str | None  # "team_a" | "team_b" | None
    px: float
    py: float
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 0.9


_DEDUP_DIST_PX = 10.0  # anti-aliasing at a circle's edge can split one blob into two adjacent contours


def _dedupe(detections: list[Detection]) -> list[Detection]:
    """Merges near-duplicate same-class detections (see module docstring
    note below) by keeping the larger-area one of each close pair."""
    kept: list[Detection] = []
    for d in detections:
        area = (d.x2 - d.x1) * (d.y2 - d.y1)
        merged = False
        for i, k in enumerate(kept):
            if k.cls != d.cls or k.team_hint != d.team_hint:
                continue
            if ((k.px - d.px) ** 2 + (k.py - d.py) ** 2) ** 0.5 < _DEDUP_DIST_PX:
                k_area = (k.x2 - k.x1) * (k.y2 - k.y1)
                if area > k_area:
                    kept[i] = d
                merged = True
                break
        if not merged:
            kept.append(d)
    return kept


def detect_frame(frame_bgr: np.ndarray, frame_idx: int) -> list[Detection]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    detections: list[Detection] = []
    for color_name, (lo, hi) in _COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 4:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            x, y, w, h = cv2.boundingRect(c)
            cls = _CLASS_FOR_COLOR[color_name]
            team_hint = color_name if cls == "player" else None
            detections.append(Detection(frame_idx, cls, team_hint, cx, cy, x, y, x + w, y + h))
    return _dedupe(detections)
