"""Renders a bird's-eye video of a pipeline run: tracked player/ball
positions, on-screen labels for detected events as they happen, and a
banner when Module A raises a review alert. This is a demo/reporting
utility on top of the existing pipeline, not a new architectural layer --
CLAUDE.md's four layers and three modules are unchanged.

Reuses the pitch-drawing code from synthetic_clip.py rather than
duplicating it, since perception's "color" backend positions are in the
same meter coordinate system the synthetic clip was rendered in.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from src.perception.synthetic_clip import (
    BALL_COLOR_BGR, FPS, FRAME_H, FRAME_W, PX_PER_M, REF_COLOR_BGR,
    TEAM_A_COLOR_BGR, TEAM_B_COLOR_BGR, pitch_background,
)
from src.run_pipeline import run_pipeline

PLAYER_RADIUS_PX = 7
BALL_RADIUS_PX = 4
EVENT_DISPLAY_S = 1.3  # how long an event caption stays on screen

_TEAM_COLOR = {"team_a": TEAM_A_COLOR_BGR, "team_b": TEAM_B_COLOR_BGR}

_EVENT_LABEL = {
    "pass": "PASS", "turnover": "TURNOVER", "shot": "SHOT",
    "foul": "FOUL (unflagged)", "goal": "GOAL",
}
_EVENT_COLOR = {
    "pass": (200, 200, 200), "turnover": (0, 165, 255), "shot": (255, 255, 0),
    "foul": (0, 0, 255), "goal": (0, 255, 0),
}


def _color_for(cls: str, team: str | None) -> tuple:
    if cls == "ball":
        return BALL_COLOR_BGR
    if cls == "referee":
        return REF_COLOR_BGR
    return _TEAM_COLOR.get(team, (200, 200, 200))


def render_demo(video_path: str = "data/raw/synthetic_match_clip.mp4",
                 out_path: str = "reports/figures/pipeline_demo.mp4") -> Path:
    result = run_pipeline(video_path, backend="color")
    df = result["metrics"]["player_time_df"]
    events = result["events"]
    alerts = result["review_alerts"]
    alert_time_s = min((a["goal_event"]["time_s"] for a in alerts), default=None)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    background = pitch_background()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (FRAME_W, FRAME_H))

    n_frames = int(df["frame"].max()) + 1
    for frame_idx in range(n_frames):
        frame = background.copy()
        t_s = frame_idx / FPS

        for _, row in df[df["frame"] == frame_idx].iterrows():
            px, py = int(row["x"] * PX_PER_M), int(row["y"] * PX_PER_M)
            radius = BALL_RADIUS_PX if row["cls"] == "ball" else PLAYER_RADIUS_PX
            cv2.circle(frame, (px, py), radius, _color_for(row["cls"], row["team"]), -1)

        y_offset = 20
        for e in events:
            if e["time_s"] <= t_s <= e["time_s"] + EVENT_DISPLAY_S:
                label = _EVENT_LABEL.get(e["type"], e["type"].upper())
                color = _EVENT_COLOR.get(e["type"], (255, 255, 255))
                cv2.putText(frame, f"{label}  t={e['time_s']:.1f}s", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                y_offset += 22

        if alert_time_s is not None and t_s >= alert_time_s:
            cv2.rectangle(frame, (0, FRAME_H - 30), (FRAME_W, FRAME_H), (0, 0, 150), -1)
            cv2.putText(frame, "REVIEW ALERT: goal follows an unflagged foul",
                        (10, FRAME_H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)
    writer.release()
    return out_path


if __name__ == "__main__":
    path = render_demo()
    print(f"Wrote demo video to {path}")
