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
from src.perception.synthetic_clip import (
    BALL_COLOR_BGR, FPS, FRAME_H, FRAME_W, PX_PER_M, REF_COLOR_BGR,
    TEAM_A_COLOR_BGR, TEAM_B_COLOR_BGR, pitch_background,
)
from src.run_pipeline import run_pipeline

# Dim gray, distinct from both team colors, referee yellow, and ball orange --
# a "non_player" (crowd/sideline/bench, see src/perception/pipeline.py's
# pitch-boundary filter) drawn in its own color rather than silently
# inheriting a team color it no longer represents.
NON_PLAYER_COLOR_BGR = (130, 130, 130)
_CLS_LABEL = {"player": "PLAYER", "referee": "REFEREE", "non_player": "NON-PLAYER"}

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
    return _TEAM_COLOR.get(team, (200, 200, 200))


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

        rows = by_frame.get(frame_idx)
        if rows is not None:
            for _, row in rows.iterrows():
                color = _color_for(row["cls"], row["team"])
                x1, y1, x2, y2 = int(row["box_x1"]), int(row["box_y1"]), int(row["box_x2"]), int(row["box_y2"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
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
