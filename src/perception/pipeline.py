"""Layer 1 entry point: reads a video, runs detection + team ID + tracking +
calibration per frame, and returns a per-frame per-object position table --
the single artifact every downstream layer consumes. See CLAUDE.md section 4.

Backend selection: real broadcast footage should use YOLOv8 (`backend="yolo"`);
the synthetic test clip uses the HSV color detector (`backend="color"`) since
pretrained COCO YOLO does not recognize painted circles as people/balls. Both
backends produce the same output schema so Layers 2-4 don't care which ran.
"""
from __future__ import annotations

import cv2
import pandas as pd

from src.perception import color_detector, team_id, yolo_detector
from src.perception.calibration import PitchCalibrator
from src.perception.tracker import CentroidTracker


def _run_color_backend(video_path: str, calibrator: PitchCalibrator) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    tracker = CentroidTracker()
    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = color_detector.detect_frame(frame, frame_idx)
        det_dicts = []
        for d in dets:
            x_m, y_m = calibrator.pixel_to_pitch(d.px, d.py)
            det_dicts.append({
                "cls": d.cls, "team": d.team_hint, "x": x_m, "y": y_m, "conf": d.conf,
            })
        tracked = tracker.update(det_dicts)
        for t in tracked:
            rows.append({
                "frame": frame_idx, "time_s": frame_idx / fps, "track_id": t["track_id"],
                "cls": t["cls"], "team": t["team"], "x": t["x"], "y": t["y"], "conf": t["conf"],
            })
        frame_idx += 1
    cap.release()
    return pd.DataFrame(rows)


def _run_yolo_backend(video_path: str, calibrator: PitchCalibrator) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tracker = CentroidTracker()
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
            cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
            x_m, y_m = calibrator.pixel_to_pitch(cx, cy)
            if b.cls == "person":
                team = f"team_{'a' if team_labels[person_i] == 0 else 'b'}"
                person_i += 1
                det_dicts.append({"cls": "player", "team": team, "x": x_m, "y": y_m, "conf": b.conf})
            else:
                det_dicts.append({"cls": "ball", "team": None, "x": x_m, "y": y_m, "conf": b.conf})

        tracked = tracker.update(det_dicts)
        for t in tracked:
            rows.append({
                "frame": frame_idx, "time_s": frame_idx / fps, "track_id": t["track_id"],
                "cls": t["cls"], "team": t["team"], "x": t["x"], "y": t["y"], "conf": t["conf"],
            })
        frame_idx += 1
    cap.release()
    return pd.DataFrame(rows)


def run_perception(video_path: str, backend: str = "color") -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if backend == "color":
        from src.perception.synthetic_clip import PX_PER_M
        calibrator = PitchCalibrator.identity_scale(PX_PER_M)
        return _run_color_backend(video_path, calibrator)
    elif backend == "yolo":
        calibrator = PitchCalibrator.placeholder_for_frame_size(frame_w, frame_h)
        return _run_yolo_backend(video_path, calibrator)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


if __name__ == "__main__":
    df = run_perception("data/raw/synthetic_match_clip.mp4", backend="color")
    print(df.head(20))
    print(f"\n{len(df)} rows, {df['track_id'].nunique()} unique tracks")
