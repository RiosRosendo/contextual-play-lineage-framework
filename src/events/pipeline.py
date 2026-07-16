"""Layer 3 entry point: combines possession-derived events (pass, turnover,
shot, goal) with the foul detector's output into one chronological event
list -- the the project spec section 4 output ("list of events with timestamp,
type, players involved, location").
"""
from __future__ import annotations

from src.events import pose_signals
from src.events.foul_detector.detect import run_foul_detection
from src.events.possession_events import detect_goal_events, detect_possession_events


def run_events(metrics_result: dict, external_goal_events: list[dict] | None = None) -> list[dict]:
    """`external_goal_events`, when given, replaces `detect_goal_events`'s
    own ball-position-crossing heuristic as the source of "goal" events --
    e.g. a match's official goal timestamp (see
    `possession_events.official_goal_event`), which doesn't depend on
    calibration succeeding around the goal moment the way the heuristic
    does. The heuristic remains the default/fallback for footage with no
    external goal source."""
    player_time_df = metrics_result["player_time_df"]
    possession_df = metrics_result["possession_df"]

    events = []
    events += detect_possession_events(possession_df, player_time_df)
    events += external_goal_events if external_goal_events else detect_goal_events(player_time_df)

    foul_events = run_foul_detection(player_time_df)
    # Keypoint-level contact types (hand_to_face / elbow_to_body /
    # shirt_pull / leg_contact) annotate each foul candidate with what
    # body parts actually met -- richer context for downstream foul
    # reasoning than box proximity alone. No-op (empty lists) when the
    # clip has no pose columns (e.g. the color backend).
    contact_events = pose_signals.contact_type_events(player_time_df)
    pose_signals.annotate_foul_contact_types(foul_events, contact_events)
    events += foul_events

    events += pose_signals.handball_events(player_time_df)
    return sorted(events, key=lambda e: e["time_s"])


if __name__ == "__main__":
    from src.perception.pipeline import run_perception
    from src.metrics.pipeline import run_metrics

    perception_df = run_perception("data/raw/synthetic_match_clip.mp4", backend="color")
    metrics = run_metrics(perception_df)
    events = run_events(metrics)
    for e in events:
        print(f"{e['time_s']:.2f}s  {e['type']:10s}  {e}")
