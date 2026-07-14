"""Layer 2 tactical/spatial metrics: possession, heatmaps, formation shape.
Derived mathematically from Layer 1 positions, per CLAUDE.md section 4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def possession_by_frame(df: pd.DataFrame) -> pd.DataFrame:
    """For each frame, the team whose player is nearest the ball is treated
    as the possessing team -- a simplified proxy for real possession logic
    (touch detection), per the section 3 "simplified formulas first"
    philosophy."""
    rows = []
    for frame, group in df.groupby("frame"):
        ball = group[group["cls"] == "ball"]
        players = group[group["cls"] == "player"]
        if ball.empty or players.empty:
            continue
        bx, by = ball.iloc[0]["x"], ball.iloc[0]["y"]
        dists = np.sqrt((players["x"] - bx) ** 2 + (players["y"] - by) ** 2)
        nearest = players.loc[dists.idxmin()]
        rows.append({
            "frame": frame, "time_s": ball.iloc[0]["time_s"],
            "possessing_team": nearest["team"], "nearest_track_id": nearest["track_id"],
            "dist_to_ball_m": float(dists.min()),
        })
    return pd.DataFrame(rows)


def possession_percentage(possession_df: pd.DataFrame) -> dict:
    if possession_df.empty:
        return {}
    counts = possession_df["possessing_team"].value_counts(normalize=True) * 100
    return counts.round(1).to_dict()


def heatmap(df: pd.DataFrame, pitch_length_m: float = 105.0, pitch_width_m: float = 68.0,
            bins: tuple[int, int] = (21, 14), track_id: int | None = None,
            team: str | None = None) -> np.ndarray:
    """2D histogram of occupied positions, in pitch bins."""
    subset = df[df["cls"] == "player"]
    if track_id is not None:
        subset = subset[subset["track_id"] == track_id]
    if team is not None:
        subset = subset[subset["team"] == team]
    hist, _, _ = np.histogram2d(
        subset["x"], subset["y"],
        bins=bins, range=[[0, pitch_length_m], [0, pitch_width_m]],
    )
    return hist


def formation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per team, per frame: centroid position and spread (std dev) as a
    coarse proxy for tactical shape/compactness."""
    players = df[df["cls"] == "player"]
    return (
        players.groupby(["frame", "team"])
        .agg(centroid_x=("x", "mean"), centroid_y=("y", "mean"),
             spread_x=("x", "std"), spread_y=("y", "std"))
        .reset_index()
    )
