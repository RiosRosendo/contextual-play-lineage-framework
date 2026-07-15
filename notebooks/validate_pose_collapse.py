"""Validates the new pose-collapse contact trigger
(src/events/foul_detector/contact_candidates.py: find_pose_collapse_candidates,
merged into find_contact_candidates) against the real clips whose actual
foul moment the existing distance+speed gate missed entirely (see
PROGRESS.md, 2026-07-15 stress-test entry).

For each clip, reports:
- distance/speed-only candidate count (the old behavior, unchanged)
- combined candidate count (distance/speed + pose-collapse, merged)
- every candidate's time and which trigger(s) fired
- whether a candidate now lands near the real foul moment identified by
  direct visual inspection in the earlier entry (chelsea_burnley ~19.6s,
  swansea_manutd ~16s, southampton_liverpool ~4s; mancity_watford's real
  moment was not independently pinned down, palace_arsenal's card had no
  visually-confirmed contact time established, so those two are reported
  without a specific "did it land on the right moment" check -- just the
  before/after candidate counts, to also see whether the new trigger
  introduces obviously spurious extra candidates)

Usage:
    python -m notebooks.validate_pose_collapse
"""
from __future__ import annotations

from src.events.foul_detector.contact_candidates import (
    _distance_speed_candidates, find_contact_candidates,
)
from src.metrics.pipeline import run_metrics
from src.perception.pipeline import run_perception

JOBS = [
    {"slug": "chelsea_burnley", "path": "data/raw/soccernet/card_chelsea_burnley.mp4",
     "real_foul_time_s": 19.6, "note": "two players tangled on the ground"},
    {"slug": "swansea_manutd", "path": "data/raw/soccernet/card_swansea_manutd.mp4",
     "real_foul_time_s": 16.0, "note": "standing sliding tackle"},
    {"slug": "southampton_liverpool", "path": "data/raw/soccernet/card_southampton_liverpool.mp4",
     "real_foul_time_s": 4.0, "note": "player down, teammate/ref converging"},
    {"slug": "mancity_watford", "path": "data/raw/soccernet/card_mancity_watford.mp4",
     "real_foul_time_s": None, "note": "real moment not independently pinned down"},
    {"slug": "palace_arsenal", "path": "data/raw/soccernet/card_palace_arsenal.mp4",
     "real_foul_time_s": None, "note": "had 8 distance/speed candidates already, none near its card"},
]

NEAR_WINDOW_S = 3.0


def run_job(job: dict) -> None:
    perception_df = run_perception(job["path"], backend="yolo")
    metrics = run_metrics(perception_df)
    player_time_df = metrics["player_time_df"]

    before = _distance_speed_candidates(player_time_df)
    combined = find_contact_candidates(player_time_df)

    print(f"\n{'#' * 70}\n{job['slug']} ({job['note']})\n{'#' * 70}")
    print(f"distance/speed-only candidates: {len(before)}")
    print(f"combined (distance/speed + pose-collapse) candidates: {len(combined)}")
    for c in combined:
        print(f"  t={c['time_s']:.2f}s  {c['team_a']} vs {c['team_b']}  "
              f"closing={c['closing_speed_mps']:.1f} m/s  triggers={c['triggers']}")

    if job["real_foul_time_s"] is not None:
        near = [c for c in combined if abs(c["time_s"] - job["real_foul_time_s"]) <= NEAR_WINDOW_S]
        hit = any("pose_collapse" in c["triggers"] for c in near)
        print(f"Real foul at t~{job['real_foul_time_s']:.1f}s -- candidates within {NEAR_WINDOW_S:.0f}s: "
              f"{len(near)}; pose-collapse trigger caught it: {hit}")


if __name__ == "__main__":
    for job in JOBS:
        run_job(job)
