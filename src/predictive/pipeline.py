"""Layer 4 entry point: attaches xG to shot events, pass probability to pass
events, and computes a pitch-control snapshot at the moment of the goal --
CLAUDE.md section 4 output ("probability maps and tactical performance
metrics").
"""
from __future__ import annotations

import pandas as pd

from src.predictive.pass_probability import pass_probability
from src.predictive.pitch_control import pitch_control_grid
from src.predictive.xg import expected_goal_probability


def _players_at(player_time_df: pd.DataFrame, frame: int) -> list[tuple[str, float, float]]:
    snap = player_time_df[(player_time_df["frame"] == frame) & (player_time_df["cls"] == "player")]
    return [(row["team"], row["x"], row["y"]) for _, row in snap.iterrows()]


def run_predictive(metrics_result: dict, events: list[dict]) -> dict:
    player_time_df = metrics_result["player_time_df"]
    enriched_events = []

    for e in events:
        e = dict(e)
        if e["type"] == "shot":
            e["xg"] = expected_goal_probability(e["location"])
        elif e["type"] == "pass":
            frame_rows = player_time_df[player_time_df["time_s"] == e["time_s"]]
            from_row = frame_rows[frame_rows["track_id"] == e["from_track_id"]]
            to_row = frame_rows[frame_rows["track_id"] == e["to_track_id"]]
            if not from_row.empty and not to_row.empty:
                from_xy = (from_row.iloc[0]["x"], from_row.iloc[0]["y"])
                to_xy = (to_row.iloc[0]["x"], to_row.iloc[0]["y"])
                opp_team = "team_b" if e["team"] == "team_a" else "team_a"
                defenders = [
                    (row["x"], row["y"]) for _, row in frame_rows.iterrows()
                    if row["cls"] == "player" and row["team"] == opp_team
                ]
                e["pass_probability"] = pass_probability(from_xy, to_xy, defenders)
        enriched_events.append(e)

    goal_events = [e for e in events if e["type"] == "goal"]
    pitch_control_at_goal = None
    if goal_events:
        goal_frame = player_time_df.iloc[
            (player_time_df["time_s"] - goal_events[0]["time_s"]).abs().argsort()[:1]
        ]["frame"].iloc[0]
        pitch_control_at_goal = pitch_control_grid(_players_at(player_time_df, int(goal_frame)))

    return {"events": enriched_events, "pitch_control_at_goal": pitch_control_at_goal}


if __name__ == "__main__":
    from src.perception.pipeline import run_perception
    from src.metrics.pipeline import run_metrics
    from src.events.pipeline import run_events

    perception_df = run_perception("data/raw/synthetic_match_clip.mp4", backend="color")
    metrics = run_metrics(perception_df)
    events = run_events(metrics)
    result = run_predictive(metrics, events)
    for e in result["events"]:
        if e["type"] in ("shot", "pass"):
            print(e)
    print("\nPitch control grid at goal (shape):", result["pitch_control_at_goal"].shape)
