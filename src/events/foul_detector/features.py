"""Pre-contact feature extraction for the foul detector, per the project spec
section 4: "Extracts features from the window before contact (5+ seconds):
who touched the ball first, each player's approach speed, trajectories."

Produces two feature groups matching the two-branch architecture:
  - a fixed-length trajectory sequence (sequence-encoder branch input)
  - a short-window contact-intensity summary (video-encoder branch input;
    stands in for pixel-derived contact biomechanics until a real video
    encoder is trained -- see model.py docstring)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_S = 5.0
N_STEPS = 25  # resampled sequence length, independent of source fps
VIDEO_WINDOW_S = 1.0


def _track_series(df: pd.DataFrame, track_id: int, t_start: float, t_end: float) -> pd.DataFrame:
    sub = df[(df["track_id"] == track_id) & (df["time_s"] >= t_start) & (df["time_s"] <= t_end)]
    return sub.sort_values("time_s")


def _resample(times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    if len(times) == 0:
        return np.zeros_like(target_times)
    return np.interp(target_times, times, values)


def extract_sequence_features(player_time_df: pd.DataFrame, track_id_a: int, track_id_b: int,
                               contact_time_s: float, ball_track_id: int | None = None) -> np.ndarray:
    """Returns an (N_STEPS, 6) array: [rel_dx, rel_dy, speed_a, speed_b,
    dist_to_ball_a, dist_to_ball_b] resampled over the pre-contact window."""
    t_start = max(0.0, contact_time_s - WINDOW_S)
    target_times = np.linspace(t_start, contact_time_s, N_STEPS)

    a = _track_series(player_time_df, track_id_a, t_start, contact_time_s)
    b = _track_series(player_time_df, track_id_b, t_start, contact_time_s)
    ball = player_time_df[player_time_df["cls"] == "ball"]
    ball = ball[(ball["time_s"] >= t_start) & (ball["time_s"] <= contact_time_s)].sort_values("time_s")

    ax = _resample(a["time_s"].values, a["x"].values, target_times)
    ay = _resample(a["time_s"].values, a["y"].values, target_times)
    bx = _resample(b["time_s"].values, b["x"].values, target_times)
    by = _resample(b["time_s"].values, b["y"].values, target_times)
    speed_a = _resample(a["time_s"].values, a["speed_mps"].values, target_times)
    speed_b = _resample(b["time_s"].values, b["speed_mps"].values, target_times)
    ballx = _resample(ball["time_s"].values, ball["x"].values, target_times)
    bally = _resample(ball["time_s"].values, ball["y"].values, target_times)

    rel_dx = ax - bx
    rel_dy = ay - by
    dist_to_ball_a = np.sqrt((ax - ballx) ** 2 + (ay - bally) ** 2)
    dist_to_ball_b = np.sqrt((bx - ballx) ** 2 + (by - bally) ** 2)

    return np.stack([rel_dx, rel_dy, speed_a, speed_b, dist_to_ball_a, dist_to_ball_b], axis=1).astype(np.float32)


def extract_video_branch_features(player_time_df: pd.DataFrame, track_id_a: int, track_id_b: int,
                                   contact_time_s: float) -> np.ndarray:
    """(4,) summary of the last VIDEO_WINDOW_S before contact: max closing
    speed, min distance, mean speed of each player. Placeholder for a real
    video encoder's contact-biomechanics embedding (the internal task list)."""
    t_start = max(0.0, contact_time_s - VIDEO_WINDOW_S)
    a = _track_series(player_time_df, track_id_a, t_start, contact_time_s)
    b = _track_series(player_time_df, track_id_b, t_start, contact_time_s)
    if a.empty or b.empty:
        return np.zeros(4, dtype=np.float32)

    n = min(len(a), len(b))
    a, b = a.iloc[:n], b.iloc[:n]
    dist = np.sqrt((a["x"].values - b["x"].values) ** 2 + (a["y"].values - b["y"].values) ** 2)
    closing_speed = a["speed_mps"].values + b["speed_mps"].values

    return np.array([
        closing_speed.max() if len(closing_speed) else 0.0,
        dist.min() if len(dist) else 5.0,
        a["speed_mps"].mean(),
        b["speed_mps"].mean(),
    ], dtype=np.float32)


def first_touch_before(player_time_df: pd.DataFrame, contact_time_s: float,
                        track_id_a: int, track_id_b: int) -> int | None:
    """Which of the two players was nearest the ball earliest in the
    pre-contact window -- a simple proxy for "who touched the ball first"."""
    t_start = max(0.0, contact_time_s - WINDOW_S)
    ball = player_time_df[player_time_df["cls"] == "ball"]
    ball = ball[(ball["time_s"] >= t_start) & (ball["time_s"] <= contact_time_s)].sort_values("time_s")
    if ball.empty:
        return None
    first_ball = ball.iloc[0]
    for tid in (track_id_a, track_id_b):
        p = _track_series(player_time_df, tid, t_start, contact_time_s)
        if p.empty:
            continue
        p0 = p.iloc[0]
        dist = ((p0["x"] - first_ball["x"]) ** 2 + (p0["y"] - first_ball["y"]) ** 2) ** 0.5
        if dist < 2.0:
            return tid
    return None
