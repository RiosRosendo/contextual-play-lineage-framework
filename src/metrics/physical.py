"""Layer 2 physical metrics: speed, acceleration, distance covered. Pure math
over Layer 1's per-frame position table -- no new models, per the project spec
section 4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_physical_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Adds speed_mps, accel_mps2, and cumulative dist_m columns, computed
    per track_id via finite differences over consecutive frames."""
    df = df.sort_values(["track_id", "time_s"]).copy()

    dx = df.groupby("track_id")["x"].diff()
    dy = df.groupby("track_id")["y"].diff()
    dt = df.groupby("track_id")["time_s"].diff()

    dist_step = np.sqrt(dx**2 + dy**2)
    speed = (dist_step / dt).replace([np.inf, -np.inf], np.nan)

    df["dist_step_m"] = dist_step.fillna(0.0)
    df["speed_mps"] = speed.fillna(0.0)
    df["accel_mps2"] = (df.groupby("track_id")["speed_mps"].diff() / dt).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    df["dist_cum_m"] = df.groupby("track_id")["dist_step_m"].cumsum()
    return df


def track_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per track_id: total distance, top speed, mean speed."""
    return (
        df.groupby(["track_id", "cls", "team"])
        .agg(
            total_dist_m=("dist_step_m", "sum"),
            top_speed_mps=("speed_mps", "max"),
            mean_speed_mps=("speed_mps", "mean"),
        )
        .reset_index()
    )
