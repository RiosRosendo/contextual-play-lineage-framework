"""Full-body pose estimation (17 COCO keypoints per player) via
yolov8n-pose, run as a SECOND pass alongside the primary person/ball
detector (yolo_detector.py) -- the dual-pass design chosen after scoping
(see the dev log, 2026-07-16): the primary detector keeps sole authority
over which detections exist (its precision/recall is validated and better
during chaotic moments than the pose model's own person detector), and
this module only contributes keypoints, matched onto the primary
detections by box IoU. A primary detection with no acceptable pose match
simply gets no keypoints that frame -- detection quality is never held
hostage to pose quality.

The full 17-joint skeleton (not just ankles) is deliberate: it feeds
contact-type identification between any two players' body parts
(hand-to-face, elbow-to-body, shirt-pull), handball detection
(wrist-to-ball proximity), and pose-based analytics (sprint mechanics,
jump height) -- see src/events/pose_signals.py.

Weights note: yolov8n-pose.pt is the plain Ultralytics COCO-pretrained
pose checkpoint, auto-downloaded on first use and gitignored by *.pt like
every other weight file. No fine-tuned or restrictively-licensed data is
involved.
"""
from __future__ import annotations

import numpy as np

# COCO 17-keypoint order, used everywhere downstream. Index in this tuple ==
# row index in the (17, 3) keypoint array.
KEYPOINT_NAMES = (
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
)

# Minimum IoU between a primary person box and a pose box for the pose's
# keypoints to be attributed to that person. Deliberately moderate: the two
# models were trained separately and their boxes for the same person differ
# a bit, but anything looser risks attributing a neighbor's skeleton to the
# wrong player in crowded scenes -- exactly where correctness matters most.
POSE_MATCH_IOU = 0.4

_model = None


def _get_model():
    global _model
    if _model is None:
        print("Loading yolov8n-pose model (first call only)...")
        from ultralytics import YOLO
        _model = YOLO("yolov8n-pose.pt")
        print("Loaded yolov8n-pose model.")
    return _model


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def estimate_frame(frame_bgr: np.ndarray) -> list[dict]:
    """Runs the pose model on one frame. Returns one dict per detected
    person: {"box": (x1, y1, x2, y2), "keypoints": ndarray (17, 3)} where
    each keypoint row is (x_px, y_px, confidence)."""
    results = _get_model().predict(frame_bgr, verbose=False)[0]
    if results.keypoints is None or len(results.boxes) == 0:
        return []
    out = []
    kpts_all = results.keypoints.data.cpu().numpy()
    for box, kpts in zip(results.boxes, kpts_all):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append({"box": (x1, y1, x2, y2), "keypoints": kpts})
    return out


def associate_keypoints(person_boxes: list[tuple], pose_detections: list[dict]) -> list[np.ndarray | None]:
    """Greedy IoU matching of pose detections onto the primary detector's
    person boxes. Returns, per input person box (order preserved), the
    matched (17, 3) keypoint array or None. Each pose detection is used at
    most once (best IoU first), so two overlapping players can't both be
    handed the same skeleton."""
    if not person_boxes or not pose_detections:
        return [None] * len(person_boxes)

    scored = []
    for pi, pbox in enumerate(person_boxes):
        for qi, pose in enumerate(pose_detections):
            iou = _iou(pbox, pose["box"])
            if iou >= POSE_MATCH_IOU:
                scored.append((iou, pi, qi))
    scored.sort(reverse=True)

    result: list[np.ndarray | None] = [None] * len(person_boxes)
    used_poses: set[int] = set()
    for iou, pi, qi in scored:
        if result[pi] is not None or qi in used_poses:
            continue
        result[pi] = pose_detections[qi]["keypoints"]
        used_poses.add(qi)
    return result
