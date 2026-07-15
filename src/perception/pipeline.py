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

from src.perception import color_detector, pitch_calibration_cv, team_id, yolo_detector
from src.perception.bytetrack_lite import ByteTrackLite
from src.perception.calibration import PitchCalibrator


def _run_color_backend(video_path: str, calibrator: PitchCalibrator) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    tracker = ByteTrackLite()
    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = color_detector.detect_frame(frame, frame_idx)
        det_dicts = [
            {"cls": d.cls, "team": d.team_hint, "box": (d.x1, d.y1, d.x2, d.y2), "conf": d.conf}
            for d in dets
        ]
        tracked = tracker.update(det_dicts)
        for t in tracked:
            cx, cy = (t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2
            x_m, y_m = calibrator.pixel_to_pitch(cx, cy)
            rows.append({
                "frame": frame_idx, "time_s": frame_idx / fps, "track_id": t["track_id"],
                "cls": t["cls"], "team": t["team"], "x": x_m, "y": y_m, "conf": t["conf"],
            })
        frame_idx += 1
    cap.release()
    return pd.DataFrame(rows)


def _run_yolo_backend(video_path: str, calibrator: PitchCalibrator) -> pd.DataFrame:
    """Detection via YOLOv8 (fine-tuned if available, see yolo_detector.py)
    plus a two-stage IoU tracker inspired by ByteTrack (see
    bytetrack_lite.py's docstring for why this replaces Ultralytics' own
    ByteTrack/BoT-SORT integration -- both were buggy in this environment)."""
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
            det_dicts.append({
                "cls": b.cls, "team": team, "box": (b.x1, b.y1, b.x2, b.y2), "conf": b.conf,
            })

        tracked = tracker.update(det_dicts)
        for t in tracked:
            cx, cy = (t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2
            x_m, y_m = calibrator.pixel_to_pitch(cx, cy)
            rows.append({
                "frame": frame_idx, "time_s": frame_idx / fps, "track_id": t["track_id"],
                "cls": "player" if t["cls"] == "person" else "ball", "team": t["team"],
                "x": x_m, "y": y_m, "conf": t["conf"],
            })
        frame_idx += 1
    cap.release()
    return pd.DataFrame(rows)


def _calibrate_yolo_backend(video_path: str, frame_w: int, frame_h: int) -> PitchCalibrator:
    """Tries real keypoint-based calibration (pitch_calibration_cv.py) on
    the first frame; a single camera is assumed for the whole clip, so one
    calibration is computed and reused throughout. Falls back to the flat
    placeholder if no pitch keypoints are confidently detected."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if ok:
        boxes = yolo_detector.detect_frame(frame, 0)
        person_boxes = [(b.x1, b.y1, b.x2, b.y2) for b in boxes if b.cls == "person"]
        calib = pitch_calibration_cv.calibrate_frame(frame, person_boxes)
        if calib is not None:
            return calib
    return PitchCalibrator.placeholder_for_frame_size(frame_w, frame_h)


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
        calibrator = _calibrate_yolo_backend(video_path, frame_w, frame_h)
        return _run_yolo_backend(video_path, calibrator)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


if __name__ == "__main__":
    df = run_perception("data/raw/synthetic_match_clip.mp4", backend="color")
    print(df.head(20))
    print(f"\n{len(df)} rows, {df['track_id'].nunique()} unique tracks")
