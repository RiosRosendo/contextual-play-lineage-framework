"""Renders a video of a pipeline run with detections, events, and review
alerts overlaid, as they happen. This is a demo/reporting utility on top
of the existing pipeline, not a new architectural layer -- the project spec's
four layers and three modules are unchanged.

Two rendering modes, chosen automatically from the input clip:

- Synthetic clip (the "color" backend): a bird's-eye view drawn on the
  flat pitch background from synthetic_clip.py, since that backend's
  positions are already in the same meter coordinate system the clip was
  rendered in. This is the original demo, unchanged.
- Real broadcast footage (the "yolo" backend): boxes drawn directly on
  the real video frames (there is no clean synthetic bird's-eye
  equivalent for real footage's imperfect calibration), colored by team,
  with a per-player speed label, scene-cut boundaries flagged, and an
  accumulating per-team position heatmap as a picture-in-picture inset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.assistant.explain import assess_foul_candidate
from src.perception import scene_cut
from src.perception.pipeline import _calibrate_shot_own
from src.perception.synthetic_clip import (
    BALL_COLOR_BGR, FPS, FRAME_H, FRAME_W, PX_PER_M, REF_COLOR_BGR,
    TEAM_A_COLOR_BGR, TEAM_B_COLOR_BGR, pitch_background,
)
from src.run_pipeline import run_pipeline

# Standard COCO 17-keypoint skeleton edges (joint name pairs, matching
# src/perception/pose_estimator.py's KEYPOINT_NAMES) -- draws what the
# dual-pass pose pipeline already computes internally (kp_<name>_x/_y/_c
# columns) but was never visualized before. Only drawn for player/referee
# rows (see the drawing call site), not non_player, so a crowd detection
# correctly excluded from "player" doesn't also get a misleading skeleton.
POSE_SKELETON = (
    ("nose", "l_eye"), ("nose", "r_eye"), ("l_eye", "l_ear"), ("r_eye", "r_ear"),
    ("l_shoulder", "r_shoulder"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"), ("l_hip", "r_hip"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
)
POSE_KEYPOINT_CONF_MIN = 0.3  # matches src/events/pose_signals.py's KEYPOINT_CONF_MIN

# Dim gray, distinct from both team colors, referee yellow, and ball orange --
# a "non_player" (crowd/sideline/bench, see src/perception/pipeline.py's
# pitch-boundary filter) drawn in its own color rather than silently
# inheriting a team color it no longer represents.
NON_PLAYER_COLOR_BGR = (130, 130, 130)
# Distinct magenta -- src/perception/pipeline.py marks a "player"-cls
# detection as "low_confidence" instead of running the pitch-boundary
# player/non_player check at all, on any shot whose calibration itself
# isn't trustworthy (calib_source != "own"). Rendered in its own color
# and label rather than silently defaulting to PLAYER, so a viewer sees
# "uncertain" instead of a confidently wrong tag.
LOW_CONFIDENCE_COLOR_BGR = (200, 0, 200)
_CLS_LABEL = {
    "player": "PLAYER", "referee": "REFEREE", "non_player": "NON-PLAYER",
    "low_confidence": "LOW-CONF",
}

SYNTHETIC_CLIP_PATH = "data/raw/synthetic_match_clip.mp4"

PLAYER_RADIUS_PX = 7
BALL_RADIUS_PX = 4
EVENT_DISPLAY_S = 1.3  # how long an event caption stays on screen
CUT_FLASH_S = 0.5  # how long the scene-cut banner stays on screen after a cut

_TEAM_COLOR = {"team_a": TEAM_A_COLOR_BGR, "team_b": TEAM_B_COLOR_BGR}

_EVENT_LABEL = {
    "pass": "PASS", "turnover": "TURNOVER", "shot": "SHOT",
    "foul": "FOUL (unflagged)", "goal": "GOAL",
}
_EVENT_COLOR = {
    "pass": (200, 200, 200), "turnover": (0, 165, 255), "shot": (255, 255, 0),
    "foul": (0, 0, 255), "goal": (0, 255, 0),
}

# Heatmap inset: bins match src/metrics/tactical.py's `heatmap` default so the
# accumulated grid means the same thing as Layer 2's own heatmap.
HEATMAP_BINS = (21, 14)
HEATMAP_INSET_W, HEATMAP_INSET_H = 210, 140
PITCH_LENGTH_M, PITCH_WIDTH_M = 105.0, 68.0

# Live tactical-radar inset (2026-07-21): current frame's calibrated
# positions plotted on the same flat pitch diagram synthetic_clip.py's
# bird's-eye renderer already draws, at the same PX_PER_M scale -- reuses
# that background image and the pitch-coordinate system Layer 1 already
# projects every row into, rather than a new projection.
RADAR_INSET_W, RADAR_INSET_H = 210, 140

# Pitch-line overlay (2026-07-21, requested after Rosendo saw a Roboflow
# tutorial's "virtual field overlay"): standard 105x68m IFAB pitch
# markings, in the same world-coordinate system pitch_calibration_cv.py
# already calibrates against. Drawn by projecting these known real points
# back into pixel space via PitchCalibrator.pitch_to_pixel -- the inverse
# of the same homography every detection's (x, y) already comes from, not
# a new calibration mechanism.
_PENALTY_BOX_DEPTH_M, _PENALTY_BOX_Y0, _PENALTY_BOX_Y1 = 16.5, 13.84, 54.16
_SIX_YARD_DEPTH_M, _SIX_YARD_Y0, _SIX_YARD_Y1 = 5.5, 24.84, 43.16
_CENTER_CIRCLE_RADIUS_M = 9.15


def _pitch_line_polylines() -> list[np.ndarray]:
    L, W = PITCH_LENGTH_M, PITCH_WIDTH_M
    polylines = [
        np.array([(0, 0), (L, 0), (L, W), (0, W), (0, 0)]),
        np.array([(L / 2, 0), (L / 2, W)]),
        np.array([(0, _PENALTY_BOX_Y0), (_PENALTY_BOX_DEPTH_M, _PENALTY_BOX_Y0),
                   (_PENALTY_BOX_DEPTH_M, _PENALTY_BOX_Y1), (0, _PENALTY_BOX_Y1)]),
        np.array([(L, _PENALTY_BOX_Y0), (L - _PENALTY_BOX_DEPTH_M, _PENALTY_BOX_Y0),
                   (L - _PENALTY_BOX_DEPTH_M, _PENALTY_BOX_Y1), (L, _PENALTY_BOX_Y1)]),
        np.array([(0, _SIX_YARD_Y0), (_SIX_YARD_DEPTH_M, _SIX_YARD_Y0),
                   (_SIX_YARD_DEPTH_M, _SIX_YARD_Y1), (0, _SIX_YARD_Y1)]),
        np.array([(L, _SIX_YARD_Y0), (L - _SIX_YARD_DEPTH_M, _SIX_YARD_Y0),
                   (L - _SIX_YARD_DEPTH_M, _SIX_YARD_Y1), (L, _SIX_YARD_Y1)]),
    ]
    angles = np.linspace(0, 2 * np.pi, 32)
    polylines.append(np.stack([
        L / 2 + _CENTER_CIRCLE_RADIUS_M * np.cos(angles),
        W / 2 + _CENTER_CIRCLE_RADIUS_M * np.sin(angles),
    ], axis=1))
    return polylines


_PITCH_LINE_WORLD_POLYLINES = _pitch_line_polylines()
# A projected point this many frame-diagonals outside the visible frame is
# from a shot whose calibration, even though it nominally succeeded (see
# calib_source == "own"), is too poorly conditioned far from its fit
# region to draw sensibly (a documented limitation, see
# pitch_calibration_cv.py) -- clipped rather than left to overflow
# cv2.polylines' int32 coordinates.
_OVERLAY_CLIP_MARGIN_DIAGONALS = 3.0


def _draw_pitch_overlay(frame: np.ndarray, calibrator, frame_w: int, frame_h: int) -> None:
    diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
    margin = _OVERLAY_CLIP_MARGIN_DIAGONALS * diag
    overlay = frame.copy()
    for poly_world in _PITCH_LINE_WORLD_POLYLINES:
        pts = [calibrator.pitch_to_pixel(x, y) for x, y in poly_world]
        if any(abs(px) > margin or abs(py) > margin for px, py in pts):
            continue  # this line's own points are too far outside a sane range to trust
        pts_arr = np.array(pts, dtype=np.int32)
        is_closed = tuple(poly_world[0]) == tuple(poly_world[-1]) or len(poly_world) > 4
        cv2.polylines(overlay, [pts_arr], isClosed=is_closed, color=(255, 255, 255),
                       thickness=2, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)


def _tactical_radar_inset(rows: pd.DataFrame) -> np.ndarray:
    """Current frame's calibrated player/ball positions on the flat pitch
    diagram -- a live snapshot, unlike `_heatmap_inset`'s accumulated
    density. Reuses `pitch_background()` and PX_PER_M directly (see the
    module-level comment above `RADAR_INSET_W`)."""
    radar = pitch_background().copy()
    if rows is not None:
        for _, row in rows.iterrows():
            if row["cls"] not in ("player", "referee", "ball") or pd.isna(row["x"]) or pd.isna(row["y"]):
                continue
            px, py = int(row["x"] * PX_PER_M), int(row["y"] * PX_PER_M)
            if not (0 <= px < radar.shape[1] and 0 <= py < radar.shape[0]):
                continue
            radius = BALL_RADIUS_PX if row["cls"] == "ball" else PLAYER_RADIUS_PX
            cv2.circle(radar, (px, py), radius, _color_for(row["cls"], row["team"]), -1)
    radar = cv2.resize(radar, (RADAR_INSET_W, RADAR_INSET_H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(radar, (0, 0), (RADAR_INSET_W - 1, RADAR_INSET_H - 1), (255, 255, 255), 1)
    cv2.putText(radar, "live positions", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (255, 255, 255), 1, cv2.LINE_AA)
    return radar


def _print_render_progress(frame_idx: int, n_frames: int) -> None:
    step = max(1, n_frames // 10)
    if (frame_idx + 1) % step == 0 or frame_idx + 1 == n_frames:
        pct = 100 * (frame_idx + 1) / n_frames if n_frames else 0
        print(f"  rendering: {frame_idx + 1}/{n_frames} frames ({pct:.0f}%)")


def _color_for(cls: str, team: str | None) -> tuple:
    if cls == "ball":
        return BALL_COLOR_BGR
    if cls == "referee":
        return REF_COLOR_BGR
    if cls == "non_player":
        return NON_PLAYER_COLOR_BGR
    if cls == "low_confidence":
        return LOW_CONFIDENCE_COLOR_BGR
    return _TEAM_COLOR.get(team, (200, 200, 200))


def _draw_skeleton(frame: np.ndarray, row: pd.Series, color: tuple) -> None:
    """Draws POSE_SKELETON's lines between this row's kp_<name>_x/_y
    columns, skipping any joint below POSE_KEYPOINT_CONF_MIN or any row
    with no pose columns at all (e.g. a clip run before the dual-pass pose
    pipeline existed, or a ball row)."""
    if "kp_nose_x" not in row.index:
        return
    for a, b in POSE_SKELETON:
        ca, cb = row.get(f"kp_{a}_c"), row.get(f"kp_{b}_c")
        if pd.isna(ca) or pd.isna(cb) or ca < POSE_KEYPOINT_CONF_MIN or cb < POSE_KEYPOINT_CONF_MIN:
            continue
        pa = (int(row[f"kp_{a}_x"]), int(row[f"kp_{a}_y"]))
        pb = (int(row[f"kp_{b}_x"]), int(row[f"kp_{b}_y"]))
        cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)


def _draw_events_and_alert(frame: np.ndarray, t_s: float, events: list[dict],
                            alert_time_s: float | None, frame_w: int, frame_h: int) -> None:
    y_offset = 20
    for e in events:
        if e["time_s"] <= t_s <= e["time_s"] + EVENT_DISPLAY_S:
            label = _EVENT_LABEL.get(e["type"], e["type"].upper())
            if e["type"] == "foul" and e.get("triggers"):
                label += f" [{'+'.join(e['triggers'])}]"
            if e["type"] == "foul" and e.get("severity"):
                label += f" severity={e['severity']}"
            color = _EVENT_COLOR.get(e["type"], (255, 255, 255))
            cv2.putText(frame, f"{label}  t={e['time_s']:.1f}s", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            y_offset += 22

    if alert_time_s is not None and t_s >= alert_time_s:
        cv2.rectangle(frame, (0, frame_h - 30), (frame_w, frame_h), (0, 0, 150), -1)
        cv2.putText(frame, "REVIEW ALERT: goal follows an unflagged foul",
                    (10, frame_h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _render_synthetic_birdseye(video_path: str, out_path: Path) -> Path:
    result = run_pipeline(video_path, backend="color")
    df = result["metrics"]["player_time_df"]
    events = result["events"]
    alerts = result["review_alerts"]
    alert_time_s = min((a["goal_event"]["time_s"] for a in alerts), default=None)

    background = pitch_background()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (FRAME_W, FRAME_H))

    n_frames = int(df["frame"].max()) + 1
    print(f"Rendering synthetic bird's-eye demo: {n_frames} frames...")
    for frame_idx in range(n_frames):
        frame = background.copy()
        t_s = frame_idx / FPS

        for _, row in df[df["frame"] == frame_idx].iterrows():
            px, py = int(row["x"] * PX_PER_M), int(row["y"] * PX_PER_M)
            radius = BALL_RADIUS_PX if row["cls"] == "ball" else PLAYER_RADIUS_PX
            cv2.circle(frame, (px, py), radius, _color_for(row["cls"], row["team"]), -1)

        _draw_events_and_alert(frame, t_s, events, alert_time_s, FRAME_W, FRAME_H)
        writer.write(frame)
        _print_render_progress(frame_idx, n_frames)
    writer.release()
    print(f"Wrote {n_frames} frames to {out_path}")
    return out_path


def _heatmap_inset(team_bins: dict[str, np.ndarray]) -> np.ndarray:
    """Combines each team's accumulated position grid into one small color
    image: team_a's density in the red channel, team_b's in the blue
    channel, normalized independently so one team's higher activity
    doesn't wash out the other's."""
    inset = np.zeros((HEATMAP_BINS[1], HEATMAP_BINS[0], 3), dtype=np.uint8)
    if "team_a" in team_bins and team_bins["team_a"].max() > 0:
        norm = (team_bins["team_a"] / team_bins["team_a"].max() * 255).astype(np.uint8)
        inset[:, :, 2] = norm.T  # red channel; .T since histogram2d shape is (nx, ny)
    if "team_b" in team_bins and team_bins["team_b"].max() > 0:
        norm = (team_bins["team_b"] / team_bins["team_b"].max() * 255).astype(np.uint8)
        inset[:, :, 0] = norm.T  # blue channel
    inset = cv2.resize(inset, (HEATMAP_INSET_W, HEATMAP_INSET_H), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(inset, (0, 0), (HEATMAP_INSET_W - 1, HEATMAP_INSET_H - 1), (255, 255, 255), 1)
    cv2.putText(inset, "positions (accum.)", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (255, 255, 255), 1, cv2.LINE_AA)
    return inset


def _render_real_overlay(video_path: str, out_path: Path, result: dict | None = None) -> Path:
    if result is None:
        result = run_pipeline(video_path, backend="yolo")
    df = result["metrics"]["player_time_df"]
    events = result["events"]
    alerts = result["review_alerts"]
    alert_time_s = min((a["goal_event"]["time_s"] for a in alerts), default=None)

    # Severity is a Module C judgment (assess_foul_candidate), not part of
    # run_events's own output -- computed once per foul event here rather
    # than per frame, so the caption can show it alongside the trigger(s).
    for e in events:
        if e["type"] == "foul" and "closing_speed_mps" in e:
            e["severity"] = assess_foul_candidate(e)["severity"]

    shots = scene_cut.split_into_shots(video_path)
    cut_frames = {start for start, _ in shots[1:]}  # shot 0's start (frame 0) isn't a "cut"

    # Pitch-line overlay (2026-07-21): each shot's OWN calibration attempt
    # is recomputed here (cheap -- one frame's keypoint detection per shot,
    # not a second full pipeline pass) purely to decide whether to draw --
    # mirrors this function's existing pattern of independently recomputing
    # `scene_cut.split_into_shots` rather than threading it through
    # `result`. Only drawn when a shot's OWN calibration succeeds
    # (matches `calib_source == "own"` in `df`) -- a shot relying on
    # `fallback_prev_shot` or the flat `placeholder` gets no overlay at
    # all, since either would draw a confidently wrong pitch, the same
    # failure mode the calibration plausibility gate already guards
    # against for metrics.
    shot_calibrators = []
    for start_frame, end_frame in shots:
        shot_calibrators.append((start_frame, end_frame, _calibrate_shot_own(video_path, start_frame)))

    def _calibrator_for_frame(frame_idx: int):
        for start_frame, end_frame, calibrator in shot_calibrators:
            if start_frame <= frame_idx < end_frame:
                return calibrator
        return None

    team_bins = {team: np.zeros(HEATMAP_BINS) for team in df["team"].dropna().unique()}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cut_flash_frames = max(1, int(round(CUT_FLASH_S * fps)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, frame_h))

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Rendering real-footage overlay demo: {n_frames} frames...")
    by_frame = {frame_idx: rows for frame_idx, rows in df.groupby("frame")}
    last_cut_frame = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_s = frame_idx / fps

        if frame_idx in cut_frames:
            last_cut_frame = frame_idx

        shot_calibrator = _calibrator_for_frame(frame_idx)
        if shot_calibrator is not None:
            _draw_pitch_overlay(frame, shot_calibrator, frame_w, frame_h)

        rows = by_frame.get(frame_idx)
        if rows is not None:
            for _, row in rows.iterrows():
                color = _color_for(row["cls"], row["team"])
                x1, y1, x2, y2 = int(row["box_x1"]), int(row["box_y1"]), int(row["box_x2"]), int(row["box_y2"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if row["cls"] in ("player", "referee"):
                    _draw_skeleton(frame, row, color)
                cls_label = _CLS_LABEL.get(row["cls"])
                if cls_label is not None:
                    if pd.notna(row["team"]):
                        cls_label += f" {row['team']}"
                    cv2.putText(frame, cls_label, (x1, max(0, y1 - 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                if row["cls"] == "player":
                    cv2.putText(frame, f"{row['speed_mps']:.1f} m/s", (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                    if row["team"] in team_bins:
                        ix = min(HEATMAP_BINS[0] - 1, max(0, int(row["x"] / PITCH_LENGTH_M * HEATMAP_BINS[0])))
                        iy = min(HEATMAP_BINS[1] - 1, max(0, int(row["y"] / PITCH_WIDTH_M * HEATMAP_BINS[1])))
                        team_bins[row["team"]][ix, iy] += 1

        _draw_events_and_alert(frame, t_s, events, alert_time_s, frame_w, frame_h)

        if last_cut_frame is not None and frame_idx - last_cut_frame < cut_flash_frames:
            cv2.rectangle(frame, (0, 0), (frame_w - 1, frame_h - 1), (0, 255, 255), 6)
            cv2.putText(frame, "SCENE CUT", (frame_w // 2 - 70, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        inset = _heatmap_inset(team_bins)
        ih, iw = inset.shape[:2]
        frame[10:10 + ih, frame_w - 10 - iw:frame_w - 10] = inset

        # Live tactical-radar inset (2026-07-21) -- bottom-left, so it
        # doesn't collide with the accumulated-heatmap inset above.
        radar = _tactical_radar_inset(rows)
        rh, rw = radar.shape[:2]
        frame[frame_h - 10 - rh:frame_h - 10, 10:10 + rw] = radar

        writer.write(frame)
        _print_render_progress(frame_idx, n_frames)
        frame_idx += 1
    cap.release()
    writer.release()
    print(f"Wrote {frame_idx} frames to {out_path}")
    return out_path


def render_demo(video_path: str = SYNTHETIC_CLIP_PATH, out_path: str | None = None,
                 result: dict | None = None) -> Path:
    """`result`, if given, must be a prior `run_pipeline(video_path, backend="yolo")`
    return value -- lets a caller inspect events/alerts (e.g. to print a
    summary) without paying for a second full pipeline run just to render
    the video from the same data. Ignored for the synthetic clip."""
    if out_path is None:
        out_path = "reports/figures/pipeline_demo.mp4" if video_path == SYNTHETIC_CLIP_PATH \
            else f"reports/figures/{Path(video_path).stem}_demo.mp4"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if video_path == SYNTHETIC_CLIP_PATH:
        return _render_synthetic_birdseye(video_path, out_path)
    return _render_real_overlay(video_path, out_path, result=result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", nargs="?", default=SYNTHETIC_CLIP_PATH,
                         help="Clip to render a demo for (defaults to the synthetic clip).")
    parser.add_argument("--out", default=None, help="Output video path.")
    args = parser.parse_args()

    path = render_demo(args.video_path, args.out)
    print(f"Wrote demo video to {path}")
