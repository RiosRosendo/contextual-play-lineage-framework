"""Layer 3 discrete events derived from possession continuity: passes,
turnovers, and shots. Simplified heuristics over Layer 2's per-frame
possession table, per CLAUDE.md section 3/4 -- no learned models needed for
this part.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GOAL_X_TEAM_A_ATTACKS = 105.0  # team_a attacks toward the +x goal in the synthetic script
SHOT_SPEED_THRESHOLD_MPS = 12.0
SHOT_DIST_TO_GOAL_M = 25.0
GOAL_LINE_MARGIN_M = 2.0
GOAL_MOUTH_Y = (30.34, 37.66)  # standard 7.32m goal width, centered on a 68m-wide pitch
POSSESSION_DEBOUNCE_FRAMES = 3  # min consecutive frames before a nearest-player change counts


def _goal_mouth_x(team: str) -> float:
    return GOAL_X_TEAM_A_ATTACKS if team == "team_a" else 0.0


def _debounce_nearest_track(possession_df: pd.DataFrame, min_persist: int = POSSESSION_DEBOUNCE_FRAMES) -> pd.DataFrame:
    """Layer 1's tracker can still fragment identities on a noisy frame or
    two, so the raw nearest-to-ball track_id can flicker without a real
    possession change happening. Requires a candidate
    track_id to persist for `min_persist` consecutive frames before it's
    accepted -- a debounce, not a model -- so single-frame tracker noise
    doesn't get reported as a pass/turnover event.
    """
    df = possession_df.reset_index(drop=True).copy()
    confirmed_track = df.loc[0, "nearest_track_id"]
    confirmed_team = df.loc[0, "possessing_team"]
    candidate_track = confirmed_track
    candidate_count = 0

    out_tracks, out_teams = [], []
    for _, row in df.iterrows():
        if row["nearest_track_id"] == candidate_track:
            candidate_count += 1
        else:
            candidate_track = row["nearest_track_id"]
            candidate_count = 1
        if candidate_track != confirmed_track and candidate_count >= min_persist:
            confirmed_track, confirmed_team = candidate_track, row["possessing_team"]
        out_tracks.append(confirmed_track)
        out_teams.append(confirmed_team)

    df["nearest_track_id"] = out_tracks
    df["possessing_team"] = out_teams
    return df


def detect_possession_events(possession_df: pd.DataFrame, player_time_df: pd.DataFrame) -> list[dict]:
    events = []
    if possession_df.empty:
        return events
    possession_df = _debounce_nearest_track(possession_df)

    prev_team = None
    prev_track = None
    for _, row in possession_df.iterrows():
        team, track_id, t = row["possessing_team"], row["nearest_track_id"], row["time_s"]
        if prev_team is None:
            prev_team, prev_track = team, track_id
            continue
        if track_id != prev_track:
            if team == prev_team:
                events.append({
                    "type": "pass", "time_s": t, "team": team,
                    "from_track_id": prev_track, "to_track_id": track_id,
                })
            else:
                events.append({
                    "type": "turnover", "time_s": t, "team": team,
                    "from_track_id": prev_track, "to_track_id": track_id,
                })
        prev_team, prev_track = team, track_id

    # Shot heuristic: ball moving fast and close to a goal mouth while a team
    # has possession.
    ball_df = player_time_df[player_time_df["cls"] == "ball"]
    for _, row in ball_df.iterrows():
        t = row["time_s"]
        poss = possession_df[possession_df["time_s"] == t]
        if poss.empty:
            continue
        team = poss.iloc[0]["possessing_team"]
        goal_x = _goal_mouth_x(team)
        dist_to_goal = abs(row["x"] - goal_x)
        if row["speed_mps"] > SHOT_SPEED_THRESHOLD_MPS and dist_to_goal < SHOT_DIST_TO_GOAL_M:
            events.append({
                "type": "shot", "time_s": t, "team": team,
                "location": (row["x"], row["y"]), "speed_mps": row["speed_mps"],
            })

    return sorted(events, key=lambda e: e["time_s"])


def detect_goal_events(player_time_df: pd.DataFrame) -> list[dict]:
    """A "goal" is the ball crossing within GOAL_LINE_MARGIN_M of either goal
    line, inside the goal mouth's y-range. Deliberately simple -- CLAUDE.md
    section 3 -- and only fires once per approach (the first frame it enters
    the goal-line margin) rather than debouncing multi-frame lingering."""
    ball_df = player_time_df[player_time_df["cls"] == "ball"].sort_values("time_s")
    events = []
    already_scored = False
    for _, row in ball_df.iterrows():
        near_x_max = row["x"] >= GOAL_X_TEAM_A_ATTACKS - GOAL_LINE_MARGIN_M
        near_x_min = row["x"] <= GOAL_LINE_MARGIN_M
        in_mouth = GOAL_MOUTH_Y[0] <= row["y"] <= GOAL_MOUTH_Y[1]
        if in_mouth and (near_x_max or near_x_min) and not already_scored:
            scoring_team = "team_a" if near_x_max else "team_b"
            events.append({
                "type": "goal", "time_s": row["time_s"], "team": scoring_team,
                "location": (row["x"], row["y"]),
            })
            already_scored = True
    return events
