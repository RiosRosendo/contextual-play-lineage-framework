"""Layer 3 entry point: combines possession-derived events (pass, turnover,
shot, goal) with the foul detector's output into one chronological event
list -- the CLAUDE.md section 4 output ("list of events with timestamp,
type, players involved, location").
"""
from __future__ import annotations

from src.events.foul_detector.detect import run_foul_detection
from src.events.possession_events import detect_goal_events, detect_possession_events


def run_events(metrics_result: dict) -> list[dict]:
    player_time_df = metrics_result["player_time_df"]
    possession_df = metrics_result["possession_df"]

    events = []
    events += detect_possession_events(possession_df, player_time_df)
    events += detect_goal_events(player_time_df)
    events += run_foul_detection(player_time_df)
    return sorted(events, key=lambda e: e["time_s"])


if __name__ == "__main__":
    from src.perception.pipeline import run_perception
    from src.metrics.pipeline import run_metrics

    perception_df = run_perception("data/raw/synthetic_match_clip.mp4", backend="color")
    metrics = run_metrics(perception_df)
    events = run_events(metrics)
    for e in events:
        print(f"{e['time_s']:.2f}s  {e['type']:10s}  {e}")
