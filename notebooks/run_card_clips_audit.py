"""Runs the foul-verdict-audit capability (src/assistant/explain.py) against
5 more real, isolated single-card incidents across 5 different matches (see
notebooks/find_single_card_incidents.py and
notebooks/download_soccernet_card_clips.py), to build a small aggregate
accuracy table instead of the single Sunderland-Liverpool anecdote already
validated.

Resolving the team-side gap: TeamColorAnchor's team_a/team_b are arbitrary
per-clip labels with no link to SoccerNet's home/away encoding. This script
samples the ACTUAL torso BGR color per team directly from the video frames
(same torso-crop method TeamColorAnchor itself uses) and prints it alongside
each match's real kit colors -- each confirmed by directly viewing a frame
of that match's downloaded clip (see the dev log), not assumed from general
knowledge alone. The team_a/team_b -> home/away mapping is then a manual
read of this printout (kept manual deliberately -- "a simple manual lookup
keyed by match, not a general solution", per instruction), applied in
notebooks/report_card_clips_audit.py.

Usage:
    python -m notebooks.run_card_clips_audit
"""
from __future__ import annotations

import cv2
import numpy as np

from src.assistant.explain import assess_foul_candidate
from src.events.pipeline import run_events
from src.metrics.pipeline import run_metrics
from src.perception import team_id
from src.perception.pipeline import run_perception

CARD_WINDOW_S = 5.0

# Each job's card sits at t=10s in its 20s trimmed clip (download_soccernet_card_clips.py
# uses WINDOW_BEFORE_S=10). kit_home/kit_away are real kit colors confirmed by directly
# viewing a representative frame from that match's clip.
JOBS = [
    {"slug": "chelsea_burnley", "path": "data/raw/soccernet/card_chelsea_burnley.mp4",
     "card_time_s": 10.0, "card_team": "away", "kit_home": "blue (Chelsea)", "kit_away": "claret/maroon (Burnley)"},
    {"slug": "palace_arsenal", "path": "data/raw/soccernet/card_palace_arsenal.mp4",
     "card_time_s": 10.0, "card_team": "home", "kit_home": "red/blue stripes (Crystal Palace)", "kit_away": "yellow (Arsenal)"},
    {"slug": "swansea_manutd", "path": "data/raw/soccernet/card_swansea_manutd.mp4",
     "card_time_s": 10.0, "card_team": "away", "kit_home": "white (Swansea)", "kit_away": "red (Man Utd)"},
    {"slug": "southampton_liverpool", "path": "data/raw/soccernet/card_southampton_liverpool.mp4",
     "card_time_s": 10.0, "card_team": "home", "kit_home": "red/white stripes (Southampton)", "kit_away": "dark grey/black (Liverpool)"},
    {"slug": "mancity_watford", "path": "data/raw/soccernet/card_mancity_watford.mp4",
     "card_time_s": 10.0, "card_team": "home", "kit_home": "sky blue (Man City)", "kit_away": "black (Watford)"},
]


def _sample_team_colors(perception_df, video_path: str, n_samples: int = 15) -> dict[str, np.ndarray | None]:
    cap = cv2.VideoCapture(video_path)
    colors: dict[str, list] = {"team_a": [], "team_b": []}
    players = perception_df[perception_df["cls"] == "player"]
    for team in ("team_a", "team_b"):
        rows = players[players["team"] == team]
        if rows.empty:
            continue
        sample = rows.sample(min(n_samples, len(rows)), random_state=0)
        for _, row in sample.iterrows():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
            ok, frame = cap.read()
            if not ok:
                continue
            color = team_id.torso_crop_mean_color(frame, row["box_x1"], row["box_y1"], row["box_x2"], row["box_y2"])
            colors[team].append(color)
    cap.release()
    return {team: (np.mean(vals, axis=0) if vals else None) for team, vals in colors.items()}


def run_job(job: dict) -> dict:
    perception_df = run_perception(job["path"], backend="yolo")
    metrics = run_metrics(perception_df)
    events = run_events(metrics)
    foul_events = [e for e in events if e["type"] == "foul"]

    team_colors = _sample_team_colors(perception_df, job["path"])
    near_card = [e for e in foul_events if abs(e["time_s"] - job["card_time_s"]) <= CARD_WINDOW_S]
    assessments = [assess_foul_candidate(e) for e in near_card]

    return {"job": job, "n_foul_total": len(foul_events), "near_card": near_card,
            "assessments": assessments, "team_colors": team_colors}


if __name__ == "__main__":
    for job in JOBS:
        print(f"\n{'#' * 70}\n{job['slug']}\n{'#' * 70}")
        r = run_job(job)
        tc = r["team_colors"]
        print(f"Foul candidates in clip: {r['n_foul_total']}; near card (+/-{CARD_WINDOW_S:.0f}s): {len(r['near_card'])}")
        print(f"Sampled BGR -- team_a: {tc['team_a']}, team_b: {tc['team_b']}")
        print(f"Known kits  -- home: {job['kit_home']}, away: {job['kit_away']}")
        for e, a in zip(r["near_card"], r["assessments"]):
            card_info = f"/{a['severity']} (card: {a['recommended_card']})" if a["verdict"] == "foul" else ""
            print(f"  t={e['time_s']:.2f}s  {e['team_a']} vs {e['team_b']}  "
                  f"closing={e['closing_speed_mps']:.1f} m/s  -> {a['verdict']}{card_info}")
