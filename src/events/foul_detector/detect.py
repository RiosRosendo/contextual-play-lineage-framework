"""Layer 3 foul detector entry point: scans contact candidates, extracts
pre-contact features for both branches, runs the fused classifier, and
returns foul events. `is_flagged` is always False in the skeleton (there is
no referee-whistle signal modeled yet) -- every detected foul is, by
construction, an "unflagged foul" candidate for Module A to reason about.
"""
from __future__ import annotations

import torch
import pandas as pd

from src.events.foul_detector.contact_candidates import find_contact_candidates
from src.events.foul_detector.features import (
    extract_sequence_features, extract_video_branch_features, first_touch_before,
)
from src.events.foul_detector.model import get_model

FOUL_PROBABILITY_THRESHOLD = 0.5


def run_foul_detection(player_time_df: pd.DataFrame) -> list[dict]:
    model = get_model()
    candidates = find_contact_candidates(player_time_df)
    events = []

    for c in candidates:
        seq = extract_sequence_features(player_time_df, c["track_id_a"], c["track_id_b"], c["time_s"])
        video = extract_video_branch_features(player_time_df, c["track_id_a"], c["track_id_b"], c["time_s"])
        with torch.no_grad():
            prob = model(
                torch.tensor(seq).unsqueeze(0), torch.tensor(video).unsqueeze(0),
            ).item()

        first_touch = first_touch_before(player_time_df, c["time_s"], c["track_id_a"], c["track_id_b"])
        events.append({
            "type": "foul", "time_s": c["time_s"], "location": c["location"],
            "team_a": c["team_a"], "track_id_a": c["track_id_a"],
            "team_b": c["team_b"], "track_id_b": c["track_id_b"],
            "closing_speed_mps": c["closing_speed_mps"],
            "first_touch_track_id": first_touch,
            "foul_probability": prob,
            "is_foul": prob > FOUL_PROBABILITY_THRESHOLD,
            "is_flagged": False,
        })
    return events


if __name__ == "__main__":
    from src.perception.pipeline import run_perception
    from src.metrics.pipeline import run_metrics

    perception_df = run_perception("data/raw/synthetic_match_clip.mp4", backend="color")
    metrics = run_metrics(perception_df)
    fouls = run_foul_detection(metrics["player_time_df"])
    for f in fouls:
        print(f)
