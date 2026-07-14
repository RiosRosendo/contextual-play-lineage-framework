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
    conf: float = 0.9


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
            cls = _CLASS_FOR_COLOR[color_name]
            team_hint = color_name if cls == "player" else None
            detections.append(Detection(frame_idx, cls, team_hint, cx, cy))
    return detections
