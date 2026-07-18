"""Layer 3 discrete events derived from possession continuity: passes,
turnovers, and shots. Simplified heuristics over Layer 2's per-frame
possession table, per the project spec section 3/4 -- no learned models needed for
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

# Goal-detection plausibility gate (2026-07-17): real-footage validation
# (Leicester-Man City) found `detect_goal_events` firing on a single spurious
# ball-position reading (x=-30.7, far outside any real pitch even with
# generous margin -- almost certainly a stray misdetection or a calibration
# artifact, not a real ball position) crossing the naive single-frame check
# near a stoppage (ball out of play), not an actual goal. Two independent
# guards, matching this project's established layered-defense pattern (e.g.
# MAX_CLOSING_SPEED_MPS + team + proximity for contact candidates): (1) a
# plausibility bound on the ball's own position, generous enough that a real
# goal-mouth ball position never trips it; (2) requiring the in-mouth,
# near-goal-line condition to hold for several CONSECUTIVE frames, not one,
# since a real goal has the ball lingering in/near the net, while a
# misdetection or a ball briefly crossing the byline for a corner/goal-kick
# does not. This heuristic should always be treated as inferior to an
# official goal label when one is available (see `official_goal_event` and
# `run_events`'s `external_goal_events` parameter) -- these guards only
# reduce this heuristic's own false-positive rate for clips/windows where no
# official label exists to use instead.
GOAL_PLAUSIBLE_MARGIN_M = 10.0  # ball x/y must fall within the pitch + this much padding
GOAL_MIN_SUSTAINED_FRAMES = 3


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


def official_goal_event(time_s: float, team: str | None = None) -> dict:
    """Builds a goal event from an external authoritative source (e.g. a
    match's own official goal timestamp, SoccerNet's Labels.json in this
    project) instead of `detect_goal_events`'s ball-position-crossing
    heuristic. Same schema (plus a `source` tag), so Module A's lineage
    graph treats it identically -- see the dev log for why this exists:
    goal-line geometry depends on calibration succeeding during exactly a
    goal's own choppiest broadcast seconds, which isn't itself the thing
    Module A's backward-search logic is trying to validate. `team` here is
    whatever the external source encodes (e.g. "home"/"away") -- it is
    NOT guaranteed to match this pipeline's internal `team_a`/`team_b`
    labels, which are assigned arbitrarily per clip by `TeamColorAnchor`;
    `find_review_alerts` doesn't depend on this field matching, only
    `location` and `team` are used for display/explanation."""
    return {"type": "goal", "time_s": time_s, "team": team, "location": None, "source": "official_label"}


def detect_goal_events(player_time_df: pd.DataFrame) -> list[dict]:
    """A "goal" is the ball crossing within GOAL_LINE_MARGIN_M of either goal
    line, inside the goal mouth's y-range, SUSTAINED for GOAL_MIN_SUSTAINED_FRAMES
    consecutive frames (2026-07-17, was a single-frame crossing -- see
    GOAL_PLAUSIBLE_MARGIN_M's comment for why that misfired on real footage).
    Also rejects any ball reading outside a generously padded pitch
    boundary before it's even considered, since a real ball position never
    needs to be that implausible to register as "in the goal mouth."
    Prefer an official goal label over this heuristic whenever one is
    available (`official_goal_event` / `run_events`'s `external_goal_events`)
    -- this remains a fallback for clips/windows with no such label, not the
    trusted source of truth."""
    ball_df = player_time_df[player_time_df["cls"] == "ball"].sort_values("time_s")
    events = []
    already_scored = False
    streak_team, streak_n, streak_start = None, 0, None
    for _, row in ball_df.iterrows():
        x, y = row["x"], row["y"]
        plausible = (
            -GOAL_PLAUSIBLE_MARGIN_M <= x <= GOAL_X_TEAM_A_ATTACKS + GOAL_PLAUSIBLE_MARGIN_M
            and -GOAL_PLAUSIBLE_MARGIN_M <= y <= 68.0 + GOAL_PLAUSIBLE_MARGIN_M
        )
        near_x_max = plausible and x >= GOAL_X_TEAM_A_ATTACKS - GOAL_LINE_MARGIN_M
        near_x_min = plausible and x <= GOAL_LINE_MARGIN_M
        in_mouth = plausible and GOAL_MOUTH_Y[0] <= y <= GOAL_MOUTH_Y[1]
        team = ("team_a" if near_x_max else "team_b") if (in_mouth and (near_x_max or near_x_min)) else None

        if team is not None and team == streak_team:
            streak_n += 1
        elif team is not None:
            streak_team, streak_n, streak_start = team, 1, row
        else:
            streak_team, streak_n, streak_start = None, 0, None

        if team is not None and streak_n >= GOAL_MIN_SUSTAINED_FRAMES and not already_scored:
            events.append({
                "type": "goal", "time_s": streak_start["time_s"], "team": team,
                "location": (streak_start["x"], streak_start["y"]),
            })
            already_scored = True
    return events
