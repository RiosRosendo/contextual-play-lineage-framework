"""Pass 1 of the two-pass (VAR-style) architecture (2026-07-18, see
reports/two_pass_architecture_scoping.md): a cheap, whole-clip scan over
box-only signals to flag which short windows of a clip are worth Pass 2's
much more expensive frame-by-frame pose/appearance analysis.

Real-footage validation (Leicester-Man City) found a serious, honestly-
disclosed gap in the first version of this module: it reused
`find_contact_candidates` directly, on the assumption that its two
box-only triggers (`_distance_speed_candidates`, calibrated position/
speed; `find_pose_collapse_candidates`, box aspect-ratio) would at least
flag SOMETHING near this project's own already-validated real fouls. They
did not -- zero windows on Leicester-Man City, Swansea-Man Utd, and
Chelsea-Burnley alike. Traced directly against three known real-foul
timestamps/pairs: calibrated distance between the two real participants
at their own contact moment was 2.8-13.2m (`CONTACT_DIST_M` requires
<=1.5m) -- a direct instance of this project's own long-documented finding
that calibrated position/speed becomes LEAST reliable exactly during real
contact, which is the whole reason the pose-dependent triggers
(`keypoint_contact`, `torso_fall`) were built in the first place.
Pixel-space box-CENTER distance (not calibrated) was similarly unhelpful
(0.14-0.90x average box height) -- two players in different poses during
a slide/tangle can have their box centers far apart even while their
bodies are touching.

**What DOES work, confirmed on the same three real pairs: box IoU
overlap** (0.42, 0.16, 0.14 -- all clearly nonzero). This is the same
signal `contact_candidates._has_confirmed_contact` already uses for a
DIFFERENT purpose (confirming a pose-collapse run's pairing); exposed
here as a standalone, purely pixel-space, calibration-independent Pass 1
trigger in its own right: any two opposing-team players whose tracked
boxes overlap at all is worth a closer look, full stop -- no speed
requirement, no fall requirement, no calibrated position at all.
"""
from __future__ import annotations

import pandas as pd

from src.events.foul_detector.contact_candidates import (
    _box_iou, _build_known_team_index, _nearest_known_team, find_contact_candidates,
)

# +/- around each trigger timestamp -- matches Rosendo's own "2-3s" spec
# for how wide a VAR-style review window should be.
REVIEW_WINDOW_MARGIN_S = 2.5
# Windows closer than this (after applying the margin above) get merged
# into one, so two triggers 0.5s apart don't produce two overlapping
# windows Pass 2 would redundantly re-process.
REVIEW_WINDOW_MERGE_GAP_S = 1.0
# Any nonzero overlap counts -- deliberately permissive (Pass 1's job is
# to not miss a window, not to judge whether it's a foul; a false-positive
# window just costs Pass 2 some extra, still-cheap-relative-to-the-whole-
# clip compute, but a false negative here means Pass 2 never even looks).
MIN_BOX_IOU = 0.01


def _merge_windows(times: list[float], margin_s: float, merge_gap_s: float) -> list[tuple[float, float]]:
    if not times:
        return []
    raw = sorted((max(0.0, t - margin_s), t + margin_s) for t in times)
    merged = [list(raw[0])]
    for start, end in raw[1:]:
        if start - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _box_overlap_times(player_time_df: pd.DataFrame) -> list[float]:
    """Any frame where two opposing-team players' tracked boxes overlap at
    all (IoU >= MIN_BOX_IOU) -- see this module's docstring for why this,
    not calibrated distance/speed or box-collapse, is the trigger that
    actually catches this project's own validated real fouls."""
    players = player_time_df[player_time_df["cls"].isin(("player", "low_confidence"))]
    if players.empty:
        return []
    team_index = _build_known_team_index(players)
    times = []
    for frame, group in players.groupby("frame"):
        rows = group.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                t = a["time_s"]
                team_a = _nearest_known_team(team_index, a["track_id"], t)
                team_b = _nearest_known_team(team_index, b["track_id"], t)
                if team_a is None or team_b is None or team_a == team_b:
                    continue
                box_a = (a["box_x1"], a["box_y1"], a["box_x2"], a["box_y2"])
                box_b = (b["box_x1"], b["box_y1"], b["box_x2"], b["box_y2"])
                if _box_iou(box_a, box_b) >= MIN_BOX_IOU:
                    times.append(t)
    return times


def find_review_windows(player_time_df: pd.DataFrame) -> list[tuple[float, float]]:
    """Returns a list of non-overlapping (t_start, t_end) windows worth
    Pass 2's deeper analysis, derived purely from Layer 1's own box-only
    output -- no dependency on Layer 2/3 possession or event data, so
    Layer 1's perception stage can run this scan and hand windows
    straight to Pass 2 without waiting on anything downstream. Combines
    three box-only signals: `find_contact_candidates`'s own two cheap
    triggers (calibrated distance/speed, box-aspect-ratio collapse) plus
    the box-IoU-overlap trigger above, which is the one confirmed to
    actually catch real fouls in this project's own validated clips."""
    times = [c["time_s"] for c in find_contact_candidates(player_time_df)]
    times += _box_overlap_times(player_time_df)
    return _merge_windows(times, REVIEW_WINDOW_MARGIN_S, REVIEW_WINDOW_MERGE_GAP_S)
