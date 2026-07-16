"""Validates src/perception/pitch_calibration_cv.py against real footage,
per Rosendo's request: replace the placeholder fixed homography in
calibration.py with real keypoint-based calibration, and show its actual
impact on a Layer 2 metric (not just that the code runs).

Two checks:

1. Geometric check on a real SoccerSum frame (goal-area strategy): fits a
   homography assuming the detected box line is the penalty box (16.5m --
   by construction, not an independent check, since that value is used
   directly in the fit). An attempt to also find a second, separate line
   (e.g. the 6-yard box) for a genuinely independent cross-check did not
   turn up a reliable one on this frame -- an earlier apparent success was
   actually the same line found twice via overlapping search windows, not
   a real independent feature (see the dev log). What IS a real, measured
   finding: the fitted homography is only reliable *near* the goal area it
   was calibrated from -- checked directly by transforming a point on the
   near touchline (250-350px further away) and getting wildly
   inconsistent, sometimes sign-flipped results.

2. Before/after Layer 2 speed impact on the SoccerTrack v2 clip (the only
   continuous real-motion footage available -- SoccerSum's "sequences"
   turned out not to be temporally continuous, see the dev log), comparing
   the flat placeholder calibration against the full-pitch-boundary
   strategy on the exact same clip/detections/tracks.

Usage:
    python -m notebooks.validate_pitch_calibration geometric
    python -m notebooks.validate_pitch_calibration speed
"""
from __future__ import annotations

import sys

import cv2
import pandas as pd

from src.metrics.physical import add_physical_metrics, track_summary
from src.perception import pitch_calibration_cv, team_id, yolo_detector
from src.perception.bytetrack_lite import ByteTrackLite
from src.perception.calibration import PitchCalibrator

SOCCERSUM_FRAME = "data/raw/soccersum/extracted/6114/6114_0210.jpg"
SOCCERTRACKV2_CLIP = "data/raw/soccertrackv2/117093_clip15s.mp4"


def geometric_validation() -> None:
    frame = cv2.imread(SOCCERSUM_FRAME)
    boxes = yolo_detector.detect_frame(frame, 0)
    person_boxes = [(b.x1, b.y1, b.x2, b.y2) for b in boxes if b.cls == "person"]

    goal = pitch_calibration_cv._detect_goal(frame)
    if goal is None:
        print("No goal detected -- calibration would fall back to the placeholder.")
        return
    (lx, ly), (rx, ry) = goal
    print(f"Detected goalposts at pixel ({lx},{ly}) and ({rx},{ry}).")

    calib = pitch_calibration_cv.calibrate_goal_area(frame, person_boxes)
    print("Goal-area calibration found:", calib is not None)
    print("(The 16.5m box depth is an input assumption of the fit, not an "
          "independently verified output -- see module docstring.)")
    if calib is None:
        return

    # Real, non-circular finding: locality. Check a point on the near
    # touchline, which is genuinely far (250-350px) from the 4 points the
    # homography was fit from.
    touchline_pixel = (200, 630)  # visually identified on this frame's near touchline
    wx, wy = calib.pixel_to_pitch(*touchline_pixel)
    print(f"\nLocality check: touchline pixel {touchline_pixel} -> world ({wx:.1f}, {wy:.1f}).")
    print("A real touchline point should land near lateral=+-34m at some plausible "
          "depth. This far from the fitted goal-area region, it typically doesn't -- "
          "the homography is only trustworthy near where it was calibrated.")


def _run_with_calibrator(video_path: str, calibrator: PitchCalibrator) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tracker = ByteTrackLite()
    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = yolo_detector.detect_frame(frame, frame_idx)
        torso_colors, person_boxes = [], []
        for b in boxes:
            if b.cls == "person":
                torso_colors.append(team_id.torso_crop_mean_color(frame, b.x1, b.y1, b.x2, b.y2))
                person_boxes.append(b)
        team_labels = team_id.assign_teams(torso_colors) if torso_colors else []
        det_dicts = []
        person_i = 0
        for b in boxes:
            if b.cls == "person":
                team = f"team_{'a' if team_labels[person_i] == 0 else 'b'}"
                person_i += 1
            else:
                team = None
            det_dicts.append({"cls": b.cls, "team": team, "box": (b.x1, b.y1, b.x2, b.y2), "conf": b.conf})
        tracked = tracker.update(det_dicts)
        for t in tracked:
            cx, cy = (t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2
            x_m, y_m = calibrator.pixel_to_pitch(cx, cy)
            rows.append({
                "frame": frame_idx, "time_s": frame_idx / fps, "track_id": t["track_id"],
                "cls": "player" if t["cls"] == "person" else "ball", "team": t["team"], "x": x_m, "y": y_m,
                "conf": t["conf"],
            })
        frame_idx += 1
    cap.release()
    return pd.DataFrame(rows)


def speed_before_after() -> None:
    cap = cv2.VideoCapture(SOCCERTRACKV2_CLIP)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame0 = cap.read()
    cap.release()

    placeholder = PitchCalibrator.placeholder_for_frame_size(w, h)
    boxes0 = yolo_detector.detect_frame(frame0, 0)
    person_boxes0 = [(b.x1, b.y1, b.x2, b.y2) for b in boxes0 if b.cls == "person"]
    real_calib = pitch_calibration_cv.calibrate_frame(frame0, person_boxes0)
    print("Real calibration found:", real_calib is not None)
    if real_calib is None:
        return

    for label, calibrator in [("BEFORE (placeholder)", placeholder), ("AFTER (real calibration)", real_calib)]:
        df = _run_with_calibrator(SOCCERTRACKV2_CLIP, calibrator)
        enriched = add_physical_metrics(df)
        summary = track_summary(enriched).sort_values("top_speed_mps", ascending=False)
        print(f"\n--- {label} ---")
        print(summary.head(5).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "geometric"
    {"geometric": geometric_validation, "speed": speed_before_after}[cmd]()
