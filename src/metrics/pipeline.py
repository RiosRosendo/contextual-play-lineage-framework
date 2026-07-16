"""Layer 2 entry point: takes Layer 1's per-frame position table and
produces the player-time tabular dataset (the project spec section 4 output),
plus the tactical aggregates (possession, heatmaps, formation) consumed by
Module B and Layer 3.
"""
from __future__ import annotations

import pandas as pd

from src.metrics.physical import add_physical_metrics, track_summary
from src.metrics.tactical import formation_summary, heatmap, possession_by_frame, possession_percentage


def run_metrics(perception_df: pd.DataFrame) -> dict:
    enriched = add_physical_metrics(perception_df)
    possession_df = possession_by_frame(enriched)
    return {
        "player_time_df": enriched,
        "track_summary_df": track_summary(enriched),
        "possession_df": possession_df,
        "possession_pct": possession_percentage(possession_df),
        "formation_df": formation_summary(enriched),
        "team_heatmaps": {
            team: heatmap(enriched, team=team)
            for team in enriched["team"].dropna().unique()
        },
    }


if __name__ == "__main__":
    from src.perception.pipeline import run_perception

    perception_df = run_perception("data/raw/synthetic_match_clip.mp4", backend="color")
    result = run_metrics(perception_df)
    print(result["track_summary_df"])
    print("\nPossession %:", result["possession_pct"])
