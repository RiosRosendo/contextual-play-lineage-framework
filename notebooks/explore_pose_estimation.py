"""Explores replacing the aspect-ratio pose-collapse proxy
(src/events/foul_detector/contact_candidates.py) with real per-player pose
estimation, using Ultralytics' own yolov8n-pose.pt -- same library already in
use (ultralytics.YOLO), no new dependency, just a different pretrained
checkpoint (COCO 17-keypoint format).

Investigates, on the 3 real clips with a visually-confirmed foul the current
aspect-ratio approach still misses (Chelsea-Burnley, Swansea-Man Utd,
Southampton-Liverpool -- see the dev log, 2026-07-15 entries):
1. Whether pose estimation runs at a usable speed on this CPU-only setup.
2. Detection quality on a sample frame vs. the existing plain person detector.
3. Whether keypoint-derived signals (torso angle from vertical, hip height
   drop, ankle/knee proximity between two players) would have caught any of
   the 3 previously-missed fouls -- especially Swansea-Man Utd's standing
   tackle, which involves no fall at all and so is structurally undetectable
   by any box-aspect-ratio signal, collapse-based or not.

This is exploratory only -- nothing here is wired into the pipeline.

Usage:
    python -m notebooks.explore_pose_estimation
"""
from __future__ import annotations

import math
import time

import cv2
from ultralytics import YOLO

# COCO 17-keypoint indices
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16
KEYPOINT_CONF_MIN = 0.3

JOBS = [
    {"slug": "chelsea_burnley", "path": "data/raw/soccernet/card_chelsea_burnley.mp4",
     "start_frame": 450, "end_frame": 475, "note": "real fall at ~frame 464 (t=18.56s)"},
    {"slug": "swansea_manutd", "path": "data/raw/soccernet/card_swansea_manutd.mp4",
     "start_frame": 390, "end_frame": 410, "note": "standing tackle at ~frame 400 (t=16.0s), no fall"},
    {"slug": "southampton_liverpool", "path": "data/raw/soccernet/card_southampton_liverpool.mp4",
     "start_frame": 95, "end_frame": 110, "note": "real contact at ~frame 105 (t=4.0s, right after a cut)"},
]


def _mid(kpts, i, j):
    (x1, y1, c1), (x2, y2, c2) = kpts[i], kpts[j]
    if c1 < KEYPOINT_CONF_MIN or c2 < KEYPOINT_CONF_MIN:
        return None
    return (x1 + x2) / 2, (y1 + y2) / 2


def torso_angle_deg(kpts) -> float | None:
    """Angle of the shoulder-midpoint -> hip-midpoint vector from vertical.
    ~0 deg standing upright, approaching 90 deg lying down."""
    shoulder = _mid(kpts, L_SHOULDER, R_SHOULDER)
    hip = _mid(kpts, L_HIP, R_HIP)
    if shoulder is None or hip is None:
        return None
    dx, dy = shoulder[0] - hip[0], shoulder[1] - hip[1]
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def ankle_knee_points(kpts) -> list[tuple[float, float]]:
    pts = []
    for i in (L_KNEE, R_KNEE, L_ANKLE, R_ANKLE):
        x, y, c = kpts[i]
        if c >= KEYPOINT_CONF_MIN:
            pts.append((x, y))
    return pts


def min_leg_distance(kpts_a, kpts_b) -> float | None:
    pts_a, pts_b = ankle_knee_points(kpts_a), ankle_knee_points(kpts_b)
    if not pts_a or not pts_b:
        return None
    return min(math.hypot(ax - bx, ay - by) for ax, ay in pts_a for bx, by in pts_b)


def run_job(model: YOLO, job: dict) -> None:
    print(f"\n{'#' * 70}\n{job['slug']} ({job['note']})\n{'#' * 70}")
    cap = cv2.VideoCapture(job["path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, job["start_frame"])
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    for frame_idx in range(job["start_frame"], job["end_frame"]):
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, verbose=False)[0]
        n_people = len(results.boxes)
        t_s = frame_idx / fps_video

        angles = []
        all_kpts = results.keypoints.data.tolist() if results.keypoints is not None else []
        for kpts in all_kpts:
            angle = torso_angle_deg(kpts)
            if angle is not None:
                angles.append(round(angle, 1))

        min_leg_dist = None
        for i in range(len(all_kpts)):
            for j in range(i + 1, len(all_kpts)):
                d = min_leg_distance(all_kpts[i], all_kpts[j])
                if d is not None and (min_leg_dist is None or d < min_leg_dist):
                    min_leg_dist = d

        print(f"  frame {frame_idx} t={t_s:.2f}s  people={n_people}  torso_angles={angles}  "
              f"min_leg_dist_px={min_leg_dist:.0f}" if min_leg_dist is not None else
              f"  frame {frame_idx} t={t_s:.2f}s  people={n_people}  torso_angles={angles}  min_leg_dist_px=None")
    cap.release()


def speed_and_quality_check(model: YOLO) -> None:
    cap = cv2.VideoCapture(JOBS[0]["path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 440)
    times = []
    n_detections = None
    for i in range(20):
        ok, frame = cap.read()
        t0 = time.time()
        results = model.predict(frame, verbose=False)[0]
        times.append(time.time() - t0)
        n_detections = len(results.boxes)
    cap.release()
    steady_state = times[1:]  # first call includes model warmup
    fps = 1 / (sum(steady_state) / len(steady_state))
    print(f"Speed check: first-call latency {times[0]:.2f}s (warmup), "
          f"steady-state {fps:.1f} fps over {len(steady_state)} frames.")
    print(f"Quality check: {n_detections} people detected on the last sampled frame "
          f"(compare against yolo_detector.py's plain person-detection count on the same frame).")


if __name__ == "__main__":
    print("Loading yolov8n-pose.pt...")
    model = YOLO("yolov8n-pose.pt")
    speed_and_quality_check(model)
    for job in JOBS:
        run_job(model, job)
