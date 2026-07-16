"""Finds candidate foul moments: frames where two opposing players are
within a contact distance and closing fast. This drives what windows get
fed to the two-branch classifier -- CLAUDE.md section 4 (Layer 3) says the
detector looks at the 5+ seconds before contact, so the "event" here is the
contact instant, and features.py extracts the pre-contact window around it.

Also adds a second, independent trigger (`find_pose_collapse_candidates`):
the distance+speed gate above depends on both players being tracked
normally (upright, distinguishable boxes) right through the moment of
contact -- but that's exactly when a fall/tackle most often breaks the
tracker. Real-footage stress-testing (PROGRESS.md, 2026-07-15) found 4 of 5
real single-card incidents produced ZERO contact candidates despite the
foul being clearly visible on inspection -- players tangled on the ground,
a standing tackle, a player down. A falling/tackled/tangled player's
tracked box collapses from tall-and-narrow to short-and-wide; this is a
signal already sitting in the tracked box geometry, independent of whether
the distance/speed gate also fires. `find_contact_candidates` merges both
triggers' output rather than loosening the original gate, which is kept
exactly as it was.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

CONTACT_DIST_M = 1.5
MIN_CLOSING_SPEED_MPS = 3.0
# Elite sprint speed tops out around 10-12 m/s (Usain Bolt's peak is ~12.4
# m/s); this is the SUM of both players' individual speeds, so even a rare
# head-on full-sprint collision shouldn't clear much past that. Real-footage
# validation (Sunderland vs Liverpool, see PROGRESS.md) found "closing
# speeds" of 16-46 m/s coming out of the tracker on a heavily cut-up clip --
# physically impossible for real contact, and a strong signal that a
# tracking-ID switch (not a genuine collision) produced the position jump
# that this heuristic misreads as speed. Filtered out here rather than left
# for the (untrained, illustrative-only) classifier to sort out downstream.
MAX_CLOSING_SPEED_MPS = 12.0
MIN_GAP_S = 1.0  # avoid re-flagging the same contact on consecutive frames

# Pose-collapse trigger: deliberately pixel-space, not calibrated meters --
# a fall/tackle is fundamentally an image-space phenomenon (the box
# geometry changes), and calibration reliability is exactly what's
# documented as shaky in real broadcast footage (PROGRESS.md), so tying
# this new path to it would reintroduce the same failure mode it's meant
# to work around.
POSE_BASELINE_WINDOW_S = 1.0  # how far back a track's "normal" standing aspect ratio is measured from
POSE_COLLAPSE_RATIO = 0.6  # current aspect ratio must drop below this fraction of the recent baseline
# Absolute fallback for tracks with no usable baseline -- e.g. a track born
# right after a scene cut, already mid-tangle, with no prior "standing"
# frames to compare against (found in the Southampton-Liverpool clip; see
# PROGRESS.md, 2026-07-15). Calibrated from the two real collapses measured
# so far: Chelsea-Burnley's confirmed on-ground player read 0.23, and
# Southampton-Liverpool's post-cut track started at 0.63 (already
# recovering, so likely not even its lowest point). 0.5 sits between them.
# Disclosed limitation: this is a single global constant, but "normal
# standing" aspect ratio itself varies by camera framing -- Chelsea-Burnley's
# own non-collapsed players in the same window read as low as ~0.44, close
# enough to this threshold that a tight/wide framing could misfire; see the
# validation entry in PROGRESS.md for the measured false-positive check.
POSE_ABS_COLLAPSE_RATIO = 0.5
# A single frame this far below normal is extreme enough to count as
# sufficient evidence on its own, without needing POSE_COLLAPSE_MIN_FRAMES
# consecutive frames -- added after finding the real Chelsea-Burnley
# collapse (0.23) only ever showed up for exactly one frame before the
# track was lost to occlusion, which the sustained-run requirement was
# rejecting outright regardless of the pairing/threshold relaxations
# (PROGRESS.md, 2026-07-15). 0.3 is chosen to sit clearly below
# Chelsea-Burnley's 0.23 with margin, while staying clearly above the
# extreme end (this is NOT calibrated against Southampton-Liverpool's 0.63
# reading -- that value doesn't clear the *existing* 0.5 bar either, let
# alone this stricter one; see the dev log for why that case needs a
# different fix, not a lower threshold, and remains a genuine miss here).
POSE_EXTREME_ABS_RATIO = 0.3
POSE_COLLAPSE_MIN_FRAMES = 3  # must stay collapsed this many consecutive frames -- not a single noisy frame
POSE_OVERLAP_TOLERANCE_S = 1.0  # a collapse and a nearby opponent count as "the same moment" within this gap
POSE_PROXIMITY_BOX_DIAGONALS = 2.0  # "nearby" boxes, scaled by box size so it holds at any camera distance
POSE_MERGE_WINDOW_S = 2.0  # merge a pose-collapse hit into an existing distance/speed hit for the same pair

# Real contact IS box overlap, which is exactly what TeamColorAnchor's
# occlusion check (see team_id.py, 2026-07-16) treats as untrustworthy --
# so both real players in a genuine tackle often have team=None for the
# whole contact instant, recovering a confident label only after it's
# passed. Confirmed directly (PROGRESS.md, 2026-07-16) that requiring a
# team label in the *exact* frame/window of contact was silently dropping
# real, previously-caught fouls (Man City-Watford, Crystal Palace-Arsenal)
# for this reason alone. TEAM_LOOKUP_WINDOW_S tolerates that gap: instead
# of requiring the exact instant to already be confidently classified,
# every team lookup searches this far around it for the nearest confident
# label on the SAME track.
TEAM_LOOKUP_WINDOW_S = 1.0
# For the pose-collapse trigger's opposing-player proximity check
# specifically: averaging that player's position over the full
# POSE_OVERLAP_TOLERANCE_S search window (up to ~2s total) diluted true
# nearness whenever they moved substantially across it (e.g. arriving at
# and then leaving the tangle) -- confirmed this was blocking Chelsea-
# Burnley's real collapse from pairing even after it started producing a
# qualifying run. Position stats are now averaged over this much narrower
# window, centered on the run's own timing, falling back to the full
# tolerance window only if the narrow one has no data at all for that
# player.
POSE_PROXIMITY_WINDOW_S = 0.3


def _build_known_team_index(players: pd.DataFrame) -> dict:
    """Per track_id, the sorted (time_s, team) pairs for frames where the
    team was confidently classified (team_id.TeamColorAnchor didn't
    abstain) -- built once per call so every team lookup below is a fast
    nearest-neighbor search instead of a fresh DataFrame filter."""
    known = players.dropna(subset=["team"])
    index = {}
    for track_id, g in known.groupby("track_id"):
        g = g.sort_values("time_s")
        index[track_id] = (g["time_s"].to_numpy(), g["team"].to_numpy())
    return index


def _nearest_known_team(index: dict, track_id, t: float, window_s: float = TEAM_LOOKUP_WINDOW_S) -> str | None:
    """The confidently-known team label for this track closest to time t,
    if one exists within window_s -- tolerates the exact instant being
    ambiguous/occluded (see TEAM_LOOKUP_WINDOW_S) rather than requiring
    that specific frame to already carry a confident label."""
    if track_id not in index:
        return None
    times, teams = index[track_id]
    pos = np.searchsorted(times, t)
    best_team, best_dt = None, window_s
    for i in (pos - 1, pos):
        if 0 <= i < len(times):
            dt = abs(times[i] - t)
            if dt <= best_dt:
                best_dt = dt
                best_team = teams[i]
    return best_team


def _distance_speed_candidates(player_time_df: pd.DataFrame) -> list[dict]:
    players = player_time_df[player_time_df["cls"] == "player"]
    team_index = _build_known_team_index(players)
    candidates = []
    last_flagged_t = {}

    for frame, group in players.groupby("frame"):
        rows = group.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                t = a["time_s"]
                # Nearest-known lookup, not each row's own (possibly None/NaN)
                # team value: real contact is box overlap, which is exactly
                # when TeamColorAnchor abstains, so requiring the exact frame
                # to already carry a confident label was silently dropping
                # real fouls (see TEAM_LOOKUP_WINDOW_S above).
                team_a = _nearest_known_team(team_index, a["track_id"], t)
                team_b = _nearest_known_team(team_index, b["track_id"], t)
                if team_a is None or team_b is None or team_a == team_b:
                    continue
                dist = ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5
                if dist > CONTACT_DIST_M:
                    continue
                closing_speed = a.get("speed_mps", 0.0) + b.get("speed_mps", 0.0)
                if closing_speed < MIN_CLOSING_SPEED_MPS or closing_speed > MAX_CLOSING_SPEED_MPS:
                    continue
                pair_key = tuple(sorted((a["track_id"], b["track_id"])))
                if t - last_flagged_t.get(pair_key, -999) < MIN_GAP_S:
                    continue
                last_flagged_t[pair_key] = t
                candidates.append({
                    "time_s": t, "frame": frame,
                    "track_id_a": a["track_id"], "team_a": team_a,
                    "track_id_b": b["track_id"], "team_b": team_b,
                    "location": ((a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2),
                    "closing_speed_mps": closing_speed,
                    "trigger": "distance_speed",
                })
    return candidates


def _collapse_runs(player_time_df: pd.DataFrame, team_index: dict) -> list[dict]:
    """Per track, finds aspect-ratio-collapse runs (a box going
    tall-and-narrow -> short-and-wide -- falling, being tackled, ending up
    tangled on the ground). Tiered acceptance: a run needs
    POSE_COLLAPSE_MIN_FRAMES consecutive collapsed frames to qualify UNLESS
    at least one frame in it is extreme enough (POSE_EXTREME_ABS_RATIO) to
    stand on its own -- a real collapse that's only ever observed for one
    frame before the track is lost to occlusion (see module docstring)
    would otherwise never qualify no matter how extreme that one frame is."""
    df = player_time_df[player_time_df["cls"] == "player"]
    if df.empty or "box_x1" not in df.columns:
        return []  # needs pixel boxes -- only the yolo backend carries these (see pipeline.py)

    dt = df["time_s"].diff().median()
    if not dt or dt <= 0:
        return []
    window = max(3, round(POSE_BASELINE_WINDOW_S / dt))
    min_periods = max(3, window // 2)

    runs = []
    for track_id, g in df.sort_values("frame").groupby("track_id"):
        g = g.reset_index(drop=True)
        w = g["box_x2"] - g["box_x1"]
        h = g["box_y2"] - g["box_y1"]
        aspect = (h / w.where(w > 0)).rename("aspect_ratio")
        baseline = aspect.rolling(window=window, min_periods=min_periods).median().shift(1)
        relative_collapse = (aspect < POSE_COLLAPSE_RATIO * baseline).fillna(False)
        absolute_collapse = (aspect < POSE_ABS_COLLAPSE_RATIO).fillna(False)
        extreme = (aspect < POSE_EXTREME_ABS_RATIO).fillna(False)
        collapsed = relative_collapse | absolute_collapse

        start, has_extreme = None, False
        for i, is_collapsed in enumerate(collapsed):
            if is_collapsed:
                if start is None:
                    start = i
                    has_extreme = bool(extreme.iloc[i])
                else:
                    has_extreme = has_extreme or bool(extreme.iloc[i])
            elif start is not None:
                if has_extreme or i - start >= POSE_COLLAPSE_MIN_FRAMES:
                    runs.append(_run_dict(track_id, g, start, i - 1, team_index))
                start, has_extreme = None, False
        if start is not None and (has_extreme or len(g) - start >= POSE_COLLAPSE_MIN_FRAMES):
            runs.append(_run_dict(track_id, g, start, len(g) - 1, team_index))
    return runs


def _run_dict(track_id, g: pd.DataFrame, i0: int, i1: int, team_index: dict) -> dict:
    seg = g.iloc[i0:i1 + 1]
    diag = ((seg["box_x2"] - seg["box_x1"]) ** 2 + (seg["box_y2"] - seg["box_y1"]) ** 2) ** 0.5
    start_t, end_t = float(seg["time_s"].iloc[0]), float(seg["time_s"].iloc[-1])
    return {
        "track_id": track_id, "team": _nearest_known_team(team_index, track_id, (start_t + end_t) / 2),
        "start_time_s": start_t, "end_time_s": end_t,
        "cx": float(((seg["box_x1"] + seg["box_x2"]) / 2).mean()),
        "cy": float(((seg["box_y1"] + seg["box_y2"]) / 2).mean()),
        "diag": float(diag.mean()),
        "x": float(seg["x"].mean()), "y": float(seg["y"].mean()),
        "speed_mps": float(seg["speed_mps"].mean()) if "speed_mps" in seg else 0.0,
    }


def _track_window_stats(df: pd.DataFrame, track_id, t0: float, t1: float,
                         team_index: dict, team_lookup_t: float) -> dict | None:
    """Same summary `_run_dict` computes (box center/diagonal, calibrated
    position, speed) but for a plain time window on a given track, not a
    collapse run -- used to describe a nearby opposing player who may not
    have collapsed at all (see `find_pose_collapse_candidates`).
    `team_lookup_t` is looked up separately from the [t0, t1] position
    window -- it should be the run's own timing, not this window's edges,
    since the team lookup already tolerates a gap on its own terms
    (TEAM_LOOKUP_WINDOW_S)."""
    seg = df[(df["track_id"] == track_id) & (df["time_s"] >= t0) & (df["time_s"] <= t1)]
    if seg.empty:
        return None
    diag = ((seg["box_x2"] - seg["box_x1"]) ** 2 + (seg["box_y2"] - seg["box_y1"]) ** 2) ** 0.5
    return {
        "track_id": track_id, "team": _nearest_known_team(team_index, track_id, team_lookup_t),
        "cx": float(((seg["box_x1"] + seg["box_x2"]) / 2).mean()),
        "cy": float(((seg["box_y1"] + seg["box_y2"]) / 2).mean()),
        "diag": float(diag.mean()),
        "x": float(seg["x"].mean()), "y": float(seg["y"].mean()),
        "speed_mps": float(seg["speed_mps"].mean()) if "speed_mps" in seg else 0.0,
    }


def find_pose_collapse_candidates(player_time_df: pd.DataFrame) -> list[dict]:
    """Flags a candidate whenever ONE player's box shows a sustained
    aspect-ratio collapse and ANY opposing-team player is "nearby" in
    pixel space at the same time (scaled by box size, NOT the calibrated
    meter distance -- see module docstring for why) -- independent of
    whether `_distance_speed_candidates` would also have fired for the
    same moment.

    Deliberately does NOT require the second player to also collapse:
    real-footage validation (PROGRESS.md, 2026-07-15) found this was the
    actual blocker in the one clip where a genuine collapse WAS detected
    (Chelsea-Burnley) -- only one of the two real players involved in the
    tangle showed a large-enough box-geometry change; the other stayed
    close to their own baseline despite being clearly part of the same
    contact.

    Applies the same MAX_CLOSING_SPEED_MPS plausibility cap the
    distance/speed gate uses (no MIN floor here -- unlike that gate, a
    near-zero closing speed is expected and legitimate for a pose-collapse
    candidate, e.g. two players already stationary/tangled on the ground).
    First attempt at this function (see PROGRESS.md) skipped this cap and
    filtered "nearby" by a per-row team check instead of each track's
    predominant team -- both real bugs, not a looser-by-design tradeoff:
    together they produced dozens of candidates per clip, including
    nonsensical same-team pairs and closing speeds up to tens of thousands
    of m/s, all from windows that happened to catch a tracking-ID-switch
    artifact near an unrelated collapse. Team lookups were later refined
    again (2026-07-16) from "predominant team over the exact window" to
    "nearest confidently-known team within TEAM_LOOKUP_WINDOW_S" -- see
    that constant's comment for why: real contact is box overlap, which is
    exactly when TeamColorAnchor abstains, so the exact window is often
    entirely unclassified for a genuine collision."""
    df = player_time_df[player_time_df["cls"] == "player"]
    team_index = _build_known_team_index(df)
    runs = _collapse_runs(player_time_df, team_index)
    candidates = []
    seen_pairs = set()
    for run in runs:
        if run["team"] is None:
            continue
        run_center = (run["start_time_s"] + run["end_time_s"]) / 2
        t0 = run["start_time_s"] - POSE_OVERLAP_TOLERANCE_S
        t1 = run["end_time_s"] + POSE_OVERLAP_TOLERANCE_S
        other_ids = df[
            (df["time_s"] >= t0) & (df["time_s"] <= t1) & (df["track_id"] != run["track_id"])
        ]["track_id"].unique()

        for other_id in other_ids:
            # Narrow window first (centered on the run's own timing, not the
            # full +/-1s search window) so the other player's position isn't
            # diluted by where they were well before/after the actual
            # moment; only fall back to the broader window if they have no
            # data at all that close (e.g. briefly occluded too).
            other = _track_window_stats(
                df, other_id, run_center - POSE_PROXIMITY_WINDOW_S, run_center + POSE_PROXIMITY_WINDOW_S,
                team_index, run_center,
            )
            if other is None:
                other = _track_window_stats(df, other_id, t0, t1, team_index, run_center)
            if other is None or other["team"] is None or other["team"] == run["team"]:
                continue
            dist_px = math.hypot(run["cx"] - other["cx"], run["cy"] - other["cy"])
            avg_diag = (run["diag"] + other["diag"]) / 2
            if avg_diag <= 0 or dist_px > POSE_PROXIMITY_BOX_DIAGONALS * avg_diag:
                continue
            closing_speed = run["speed_mps"] + other["speed_mps"]
            if closing_speed > MAX_CLOSING_SPEED_MPS:
                continue
            pair_key = tuple(sorted((run["track_id"], other_id)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            candidates.append({
                "time_s": run["start_time_s"],
                "track_id_a": run["track_id"], "team_a": run["team"],
                "track_id_b": other_id, "team_b": other["team"],
                "location": ((run["x"] + other["x"]) / 2, (run["y"] + other["y"]) / 2),
                "closing_speed_mps": closing_speed,
                "trigger": "pose_collapse",
            })
    return sorted(candidates, key=lambda c: c["time_s"])


def find_contact_candidates(player_time_df: pd.DataFrame) -> list[dict]:
    """Union of two independent triggers: the original distance+speed gate
    (unchanged) and the new pose-collapse gate (see
    `find_pose_collapse_candidates`). A pose-collapse hit for a pair
    already flagged by the distance/speed gate around the same time is
    merged into that same candidate (adds "pose_collapse" to its
    "triggers" list) rather than appended as a separate duplicate event
    for what is very likely the same real contact."""
    candidates = _distance_speed_candidates(player_time_df)
    for c in candidates:
        c["triggers"] = [c.pop("trigger")]

    for pc in find_pose_collapse_candidates(player_time_df):
        pair_key = tuple(sorted((pc["track_id_a"], pc["track_id_b"])))
        merged = False
        for c in candidates:
            if tuple(sorted((c["track_id_a"], c["track_id_b"]))) == pair_key \
                    and abs(c["time_s"] - pc["time_s"]) < POSE_MERGE_WINDOW_S:
                c["triggers"].append(pc.pop("trigger"))
                merged = True
                break
        if not merged:
            pc["triggers"] = [pc.pop("trigger")]
            candidates.append(pc)

    return sorted(candidates, key=lambda c: c["time_s"])
