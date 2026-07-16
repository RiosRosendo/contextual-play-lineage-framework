"""Validates the TeamColorAnchor stability fix (src/perception/team_id.py):
a team-color sample is now classified as None (untrustworthy) rather than
forced to a guess when it isn't clearly closer to one reference centroid
than the other, or when its box significantly overlaps another player's --
both meant to stop a single track's reported team flipping mid-life during
occlusion (confirmed in Chelsea-Burnley and Southampton-Liverpool; see
the dev log, 2026-07-15 entries).

For each real clip, reports per track_id how many distinct non-None team
labels it was ever given over its lifetime: 1 is stable (the fix is
working, or there was never a problem for that track), 2 means it still
flips. Specifically checks whether the two known problem clips
(Chelsea-Burnley, Southampton-Liverpool) are now stable, and reports the
same distribution for the other 4 as a regression/false-positive check --
a track being labeled None more often is an acceptable cost of "don't
guess", but a NEW flip that wasn't there before would be a regression.

Usage:
    python -m notebooks.validate_team_stability
"""
from __future__ import annotations

from src.perception.pipeline import run_perception

CLIPS = [
    "data/raw/soccernet/card_chelsea_burnley.mp4",
    "data/raw/soccernet/card_swansea_manutd.mp4",
    "data/raw/soccernet/card_southampton_liverpool.mp4",
    "data/raw/soccernet/card_mancity_watford.mp4",
    "data/raw/soccernet/card_palace_arsenal.mp4",
    "data/raw/soccernet/foul_before_goal_clip.mp4",
]


def check_clip(path: str) -> None:
    print(f"\n{'#' * 70}\n{path}\n{'#' * 70}")
    df = run_perception(path, backend="yolo")
    players = df[df["cls"] == "player"]

    n_stable, n_flipping, n_all_none = 0, 0, 0
    flips = []
    for track_id, g in players.groupby("track_id"):
        distinct_teams = g["team"].dropna().unique()
        n_none = g["team"].isna().sum()
        if len(distinct_teams) == 0:
            n_all_none += 1
        elif len(distinct_teams) == 1:
            n_stable += 1
        else:
            n_flipping += 1
            flips.append((track_id, sorted(distinct_teams), len(g), n_none))

    n_tracks = players["track_id"].nunique()
    n_none_rows = players["team"].isna().sum()
    print(f"{n_tracks} player tracks total: {n_stable} stable (1 team), "
          f"{n_flipping} still flipping (2+ teams), {n_all_none} all-None (never confidently classified)")
    print(f"{n_none_rows}/{len(players)} player-rows are None (unclassified this frame)")
    if flips:
        print("Still-flipping tracks:")
        for track_id, teams, n_rows, n_none in flips:
            print(f"  track {track_id}: teams={teams}, {n_rows} rows, {n_none} None")


if __name__ == "__main__":
    for path in CLIPS:
        check_clip(path)
