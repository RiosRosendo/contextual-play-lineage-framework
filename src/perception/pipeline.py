"""Layer 1 entry point: reads a video, runs detection + team ID + tracking +
calibration per frame, and returns a per-frame per-object position table --
the single artifact every downstream layer consumes. See the project spec section 4.

Backend selection: real broadcast footage should use YOLOv8 (`backend="yolo"`);
the synthetic test clip uses the HSV color detector (`backend="color"`) since
pretrained COCO YOLO does not recognize painted circles as people/balls. Both
backends produce the same output schema so Layers 2-4 don't care which ran.
"""
from __future__ import annotations

import cv2
import pandas as pd

from src.perception import color_detector, pitch_calibration_cv, scene_cut, team_id, yolo_detector
from src.perception.bytetrack_lite import ByteTrackLite
from src.perception.calibration import PitchCalibrator

_SHOT_TRACK_ID_STRIDE = 100_000  # keeps track_ids globally unique across shots


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


def _run_yolo_backend_shot(video_path: str, calibrator: PitchCalibrator, fps: float,
                            start_frame: int, end_frame: int, track_id_offset: int,
                            team_anchor: team_id.TeamColorAnchor, calib_source: str,
                            processed_so_far: int, total_frames: int) -> tuple[list[dict], int]:
    """Runs detection + team ID + tracking over a single shot's frame range
    only. A fresh tracker is used per shot -- track identity across a cut
    is meaningless (it's a different framing, possibly a different part of
    the pitch or a different subject entirely), so continuing the same
    tracker across a cut would silently associate unrelated detections.

    Team identity (`team_anchor`) is the opposite: it's passed in from the
    caller and shared across every shot in the clip, not recreated here --
    see TeamColorAnchor's docstring for why re-clustering blind per shot
    (or per frame) is the bug it fixes.

    `processed_so_far`/`total_frames` are only for the periodic progress
    print -- YOLO detection per frame is the slow part of this pipeline
    (visible as a long silent pause otherwise), so this prints every 100
    frames across the whole clip, not just this shot."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    tracker = ByteTrackLite()
    rows = []
    processed = processed_so_far
    for frame_idx in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        boxes = yolo_detector.detect_frame(frame, frame_idx)
        torso_colors, person_box_tuples = [], []
        for b in boxes:
            if b.cls == "person":
                torso_colors.append(team_id.torso_crop_mean_color(frame, b.x1, b.y1, b.x2, b.y2))
                person_box_tuples.append((b.x1, b.y1, b.x2, b.y2))
        team_labels = team_anchor.assign(torso_colors, person_box_tuples) if torso_colors else []

        det_dicts = []
        person_i = 0
        for b in boxes:
            if b.cls == "person":
                label = team_labels[person_i]
                team = None if label is None else f"team_{'a' if label == 0 else 'b'}"
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
                "frame": frame_idx, "time_s": frame_idx / fps, "track_id": t["track_id"] + track_id_offset,
                "cls": "player" if t["cls"] == "person" else "ball", "team": t["team"],
                "x": x_m, "y": y_m, "conf": t["conf"], "calib_source": calib_source,
                "box_x1": t["box"][0], "box_y1": t["box"][1], "box_x2": t["box"][2], "box_y2": t["box"][3],
            })

        processed += 1
        if processed % 100 == 0 or processed == total_frames:
            pct = 100 * processed / total_frames if total_frames else 0
            print(f"  perception: {processed}/{total_frames} frames ({pct:.0f}%)")
    cap.release()
    return rows, processed


def _calibrate_shot_own(video_path: str, start_frame: int) -> PitchCalibrator | None:
    """Tries real keypoint-based calibration (pitch_calibration_cv.py) on a
    shot's first frame. Returns None if no pitch keypoints are confidently
    detected -- the caller decides the fallback (nearest preceding shot's
    calibration, or the flat placeholder as a last resort); see
    `_run_yolo_backend`."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    boxes = yolo_detector.detect_frame(frame, start_frame)
    person_boxes = [(b.x1, b.y1, b.x2, b.y2) for b in boxes if b.cls == "person"]
    return pitch_calibration_cv.calibrate_frame(frame, person_boxes)


def _run_yolo_backend(video_path: str, frame_w: int, frame_h: int) -> pd.DataFrame:
    """Detection via YOLOv8 (fine-tuned if available, see yolo_detector.py)
    plus a two-stage IoU tracker inspired by ByteTrack (see
    bytetrack_lite.py's docstring for why this replaces Ultralytics' own
    ByteTrack/BoT-SORT integration -- both were buggy in this environment).

    Splits the video into shots first (scene_cut.py) -- real broadcast
    footage cuts between camera angles within seconds, and both
    calibration (one homography per clip) and tracking (identities
    assumed continuous) silently produce nonsense across a cut otherwise.
    Each shot gets its own calibration attempt and a fresh tracker -- but
    team identity (TeamColorAnchor) is deliberately shared across all
    shots, so "team_a" keeps meaning the same real jersey color for the
    whole clip.

    Calibration fallback chain, per shot: (1) the shot's own keypoint
    detection (pitch_calibration_cv.py); (2) if that fails -- expected
    often on short ~100-frame shots, which rarely give the keypoint
    detector enough frames/context to lock on -- reuse the nearest
    *preceding* shot's calibration, since consecutive broadcast shots
    frequently share the same camera framing (e.g. a cut to a close-up
    and back); (3) only if no preceding shot has calibrated yet (e.g. the
    very first shot) fall back to the flat placeholder. Each row records
    which tier produced its calibration (`calib_source`) so this can be
    audited after the fact."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    shots = scene_cut.split_into_shots(video_path)
    print(f"Perception (yolo backend): {total_frames} frames across {len(shots)} shot(s)...")
    team_anchor = team_id.TeamColorAnchor()
    last_valid_calibrator: PitchCalibrator | None = None
    rows = []
    processed = 0
    for shot_i, (start_frame, end_frame) in enumerate(shots):
        print(f"Shot {shot_i + 1}/{len(shots)} (frames {start_frame}-{end_frame})...")
        own_calibrator = _calibrate_shot_own(video_path, start_frame)
        if own_calibrator is not None:
            calibrator, calib_source = own_calibrator, "own"
            last_valid_calibrator = own_calibrator
        elif last_valid_calibrator is not None:
            calibrator, calib_source = last_valid_calibrator, "fallback_prev_shot"
        else:
            calibrator = PitchCalibrator.placeholder_for_frame_size(frame_w, frame_h)
            calib_source = "placeholder"
        shot_rows, processed = _run_yolo_backend_shot(
            video_path, calibrator, fps, start_frame, end_frame, shot_i * _SHOT_TRACK_ID_STRIDE,
            team_anchor, calib_source, processed, total_frames,
        )
        rows.extend(shot_rows)
    print(f"Perception (yolo backend) done: {processed}/{total_frames} frames processed.")
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
        return _run_yolo_backend(video_path, frame_w, frame_h)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


if __name__ == "__main__":
    df = run_perception("data/raw/synthetic_match_clip.mp4", backend="color")
    print(df.head(20))
    print(f"\n{len(df)} rows, {df['track_id'].nunique()} unique tracks")
