"""Finds candidate foul moments: frames where two opposing players are
within a contact distance and closing fast. This drives what windows get
fed to the two-branch classifier -- the project spec section 4 (Layer 3) says the
detector looks at the 5+ seconds before contact, so the "event" here is the
contact instant, and features.py extracts the pre-contact window around it.

Also adds a second, independent trigger (`find_pose_collapse_candidates`):
the distance+speed gate above depends on both players being tracked
normally (upright, distinguishable boxes) right through the moment of
contact -- but that's exactly when a fall/tackle most often breaks the
tracker. Real-footage stress-testing (the dev log, 2026-07-15) found 4 of 5
real single-card incidents produced ZERO contact candidates despite the
foul being clearly visible on inspection -- players tangled on the ground,
a standing tackle, a player down. A falling/tackled/tangled player's
tracked box collapses from tall-and-narrow to short-and-wide; this is a
signal already sitting in the tracked box geometry, independent of whether
the distance/speed gate also fires. `find_contact_candidates` merges all
triggers' output rather than loosening the original gate, which is kept
exactly as it was.

A fourth trigger (`find_keypoint_contact_candidates`, 2026-07-16) promotes
real joint-to-joint contact from the dual-pass pose skeleton
(src/events/pose_signals.py) directly into foul candidates, instead of only
ever annotating a candidate some other trigger already found. This closed
a real gap: on Swansea-Man Utd and Chelsea-Burnley, none of the three
triggers above produced any candidate at all, so the correctly-detected
real leg_contact/hand_to_face signals had no live foul_candidate to attach
to and never reached Module A or Module C.
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
# validation (Sunderland vs Liverpool, see the dev log) found "closing
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
# documented as shaky in real broadcast footage (the dev log), so tying
# this new path to it would reintroduce the same failure mode it's meant
# to work around.
POSE_BASELINE_WINDOW_S = 1.0  # how far back a track's "normal" standing aspect ratio is measured from
POSE_COLLAPSE_RATIO = 0.6  # current aspect ratio must drop below this fraction of the recent baseline
# Absolute fallback for tracks with no usable baseline -- e.g. a track born
# right after a scene cut, already mid-tangle, with no prior "standing"
# frames to compare against (found in the Southampton-Liverpool clip; see
# the dev log, 2026-07-15). Calibrated from the two real collapses measured
# so far: Chelsea-Burnley's confirmed on-ground player read 0.23, and
# Southampton-Liverpool's post-cut track started at 0.63 (already
# recovering, so likely not even its lowest point). 0.5 sits between them.
# Disclosed limitation: this is a single global constant, but "normal
# standing" aspect ratio itself varies by camera framing -- Chelsea-Burnley's
# own non-collapsed players in the same window read as low as ~0.44, close
# enough to this threshold that a tight/wide framing could misfire; see the
# validation entry in the dev log for the measured false-positive check.
POSE_ABS_COLLAPSE_RATIO = 0.5
# A single frame this far below normal is extreme enough to count as
# sufficient evidence on its own, without needing POSE_COLLAPSE_MIN_FRAMES
# consecutive frames -- added after finding the real Chelsea-Burnley
# collapse (0.23) only ever showed up for exactly one frame before the
# track was lost to occlusion, which the sustained-run requirement was
# rejecting outright regardless of the pairing/threshold relaxations
# (the dev log, 2026-07-15). 0.3 is chosen to sit clearly below
# Chelsea-Burnley's 0.23 with margin, while staying clearly above the
# extreme end (this is NOT calibrated against Southampton-Liverpool's 0.63
# reading -- that value doesn't clear the *existing* 0.5 bar either, let
# alone this stricter one; see the dev log for why that case needs a
# different fix, not a lower threshold, and remains a genuine miss here).
POSE_EXTREME_ABS_RATIO = 0.3
# How far a track's own head (box top edge) must drop below its recent
# normal position, as a fraction of its recent normal box height, to count
# as a genuine fall rather than a jump or a kick (see _collapse_runs).
# 0.3 is a moderate bar: a real fall typically drops the head most of the
# way to where the feet were (ratio approaching 1.0), while a jump moves
# the head UP (a negative ratio) and a standing kick barely moves it.
POSE_MIN_VERTICAL_DROP_RATIO = 0.3
POSE_COLLAPSE_MIN_FRAMES = 3  # must stay collapsed this many consecutive frames -- not a single noisy frame
POSE_OVERLAP_TOLERANCE_S = 1.0  # a collapse and a nearby opponent count as "the same moment" within this gap
POSE_PROXIMITY_BOX_DIAGONALS = 2.0  # "nearby" boxes, scaled by box size so it holds at any camera distance
POSE_MERGE_WINDOW_S = 2.0  # merge a pose-collapse hit into an existing distance/speed hit for the same pair

# Real contact IS box overlap, which is exactly what TeamColorAnchor's
# occlusion check (see team_id.py, 2026-07-16) treats as untrustworthy --
# so both real players in a genuine tackle often have team=None for the
# whole contact instant, recovering a confident label only after it's
# passed. Confirmed directly (the dev log, 2026-07-16) that requiring a
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

# Torso-angle fall detector: a purpose-built "is this player down" signal
# using the dual-pass pose skeleton (src/perception/pose_estimator.py)
# instead of box aspect ratio -- more direct, since it measures body
# orientation rather than inferring it from box shape (which a jump or a
# kick can also distort; see POSE_MIN_VERTICAL_DROP_RATIO above). Reuses
# the same pairing/proximity/speed-check machinery as the pose-collapse
# trigger (_pair_runs_with_opponents) -- only how a "run" is detected
# differs. Added 2026-07-16; validated once against the two known-hard
# cases (Chelsea-Burnley, Man City-Watford) per instruction, not iterated
# further regardless of outcome -- see the dev log for the result.
TORSO_KEYPOINT_CONF_MIN = 0.3
# Angle of the shoulder-midpoint -> hip-midpoint vector from vertical (0
# deg standing upright, 90 deg lying flat). The earlier per-frame
# exploration (notebooks/explore_pose_estimation.py) measured a sustained
# 74-81 deg reading across ~9 consecutive frames on Chelsea-Burnley's real
# fall -- comfortably above both bars below, with margin.
TORSO_ANGLE_FALL_DEG = 60.0  # sustained-run threshold
TORSO_ANGLE_EXTREME_DEG = 70.0  # single-frame threshold (same tiered logic as POSE_EXTREME_ABS_RATIO)
TORSO_FALL_MIN_FRAMES = 3

# Keypoint-contact trigger: promotes src/events/pose_signals.contact_type_events
# (joint-to-joint proximity between opposing players -- hand_to_face,
# elbow_to_body, shirt_pull, leg_contact) directly into first-class foul
# candidates, alongside the box-geometry triggers above. Motivation (the dev
# log, 2026-07-16): those real contact signals were already being detected
# correctly at the keypoint level, but were stranded in an isolated report --
# `annotate_foul_contact_types` only ever *annotates* an existing candidate,
# so on clips where none of the box-geometry triggers fired at all (confirmed
# on both Swansea-Man Utd and Chelsea-Burnley), the real leg_contact/
# hand_to_face signal never reached a live foul_candidate/foul event, and
# therefore never reached Module A or Module C. shirt_pull is deliberately
# excluded here (not part of what was asked to be wired in) -- it stays
# annotation-only for now.
KEYPOINT_CONTACT_TRIGGER_TYPES = ("leg_contact", "hand_to_face", "elbow_to_body")


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
    tangled on the ground) that are ALSO accompanied by the track's own
    head (box top) dropping relative to its recent normal height
    (POSE_MIN_VERTICAL_DROP_RATIO) -- the aspect-ratio change alone doesn't
    tell a fall apart from a jump or a kick, both of which change a box's
    shape quickly too without the player ever going down. Tiered
    acceptance on top of that: a run needs POSE_COLLAPSE_MIN_FRAMES
    consecutive collapsed-and-falling frames to qualify UNLESS at least
    one frame in it is extreme enough (POSE_EXTREME_ABS_RATIO) to stand on
    its own -- a real collapse that's only ever observed for one frame
    before the track is lost to occlusion (see module docstring) would
    otherwise never qualify no matter how extreme that one frame is."""
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

        # A collapsing aspect ratio alone doesn't distinguish a real fall
        # from a jump or a kick -- both change a box's shape quickly too.
        # Real-footage validation (the dev log, 2026-07-16) found exactly
        # this: a defensive wall jumping to block a free kick and a
        # penalty kick both got misread as "collapses". A genuine fall's
        # extra signature is that the player's own head (the box's top
        # edge) actually moves DOWN the frame relative to their own recent
        # normal height -- a jump moves it UP instead, and a standing kick
        # barely moves it at all.
        y1_baseline = g["box_y1"].rolling(window=window, min_periods=min_periods).median().shift(1)
        h_baseline = h.rolling(window=window, min_periods=min_periods).median().shift(1)
        vertical_drop_ratio = (g["box_y1"] - y1_baseline) / h_baseline.where(h_baseline > 0)
        is_falling = (vertical_drop_ratio > POSE_MIN_VERTICAL_DROP_RATIO).fillna(False)

        relative_collapse &= is_falling
        absolute_collapse &= is_falling
        extreme &= is_falling
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


def _torso_angle_series(g: pd.DataFrame) -> pd.Series:
    """Shoulder-midpoint -> hip-midpoint angle from vertical, per frame.
    Requires both shoulder keypoints AND both hip keypoints above
    TORSO_KEYPOINT_CONF_MIN -- NaN otherwise (matches the earlier per-frame
    exploration's stricter "both landmarks confident" requirement rather
    than falling back to a single side)."""
    needed = ("kp_l_shoulder_x", "kp_r_shoulder_x", "kp_l_hip_x", "kp_r_hip_x")
    if not all(c in g.columns for c in needed):
        return pd.Series(np.nan, index=g.index)
    sh_ok = (g["kp_l_shoulder_c"] >= TORSO_KEYPOINT_CONF_MIN) & (g["kp_r_shoulder_c"] >= TORSO_KEYPOINT_CONF_MIN)
    hip_ok = (g["kp_l_hip_c"] >= TORSO_KEYPOINT_CONF_MIN) & (g["kp_r_hip_c"] >= TORSO_KEYPOINT_CONF_MIN)
    sh_x = (g["kp_l_shoulder_x"] + g["kp_r_shoulder_x"]) / 2
    sh_y = (g["kp_l_shoulder_y"] + g["kp_r_shoulder_y"]) / 2
    hip_x = (g["kp_l_hip_x"] + g["kp_r_hip_x"]) / 2
    hip_y = (g["kp_l_hip_y"] + g["kp_r_hip_y"]) / 2
    angle = np.degrees(np.arctan2((sh_x - hip_x).abs(), (sh_y - hip_y).abs()))
    return angle.where(sh_ok & hip_ok)


def _torso_fall_runs(player_time_df: pd.DataFrame, team_index: dict) -> list[dict]:
    """Per track, finds sustained (or single-frame-extreme) torso-angle
    "lying down" runs -- the keypoint-based analog of `_collapse_runs`,
    same tiered acceptance logic, different underlying signal."""
    df = player_time_df[player_time_df["cls"] == "player"]
    if df.empty or "kp_l_shoulder_x" not in df.columns:
        return []  # needs the dual-pass pose columns (see pipeline.py/pose_estimator.py)

    runs = []
    for track_id, g in df.sort_values("frame").groupby("track_id"):
        g = g.reset_index(drop=True)
        angle = _torso_angle_series(g)
        falling = (angle > TORSO_ANGLE_FALL_DEG).fillna(False)
        extreme = (angle > TORSO_ANGLE_EXTREME_DEG).fillna(False)

        start, has_extreme = None, False
        for i, is_falling in enumerate(falling):
            if is_falling:
                if start is None:
                    start = i
                    has_extreme = bool(extreme.iloc[i])
                else:
                    has_extreme = has_extreme or bool(extreme.iloc[i])
            elif start is not None:
                if has_extreme or i - start >= TORSO_FALL_MIN_FRAMES:
                    runs.append(_run_dict(track_id, g, start, i - 1, team_index))
                start, has_extreme = None, False
        if start is not None and (has_extreme or len(g) - start >= TORSO_FALL_MIN_FRAMES):
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
    real-footage validation (the dev log, 2026-07-15) found this was the
    actual blocker in the one clip where a genuine collapse WAS detected
    (Chelsea-Burnley) -- only one of the two real players involved in the
    tangle showed a large-enough box-geometry change; the other stayed
    close to their own baseline despite being clearly part of the same
    contact.

Applies the same MAX_CLOSING_SPEED_MPS plausibility cap the
    distance/speed gate uses, but only to the OTHER player's speed, not the
    collapsing player's own (see the check itself for why -- a real
    collapse inflates the falling player's own computed speed by
    construction, so including it rejected the clearest real falls
    hardest; fixed 2026-07-16). No MIN floor here either way -- unlike the
    distance/speed gate, a near-zero speed is expected and legitimate for
    a pose-collapse candidate, e.g. two players already stationary/tangled
    on the ground.
    First attempt at this function (see the dev log) skipped this cap and
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
    return _pair_runs_with_opponents(df, team_index, runs, "pose_collapse")


def find_torso_fall_candidates(player_time_df: pd.DataFrame) -> list[dict]:
    """Same idea as `find_pose_collapse_candidates`, but the "is this
    player down" signal is torso angle from the dual-pass pose skeleton
    (`_torso_fall_runs`) instead of box aspect ratio -- see
    TORSO_ANGLE_FALL_DEG's comment for why this is a more direct signal.
    Shares the exact same pairing/proximity/other-player-speed-only check
    as the pose-collapse trigger (`_pair_runs_with_opponents`)."""
    df = player_time_df[player_time_df["cls"] == "player"]
    team_index = _build_known_team_index(df)
    runs = _torso_fall_runs(player_time_df, team_index)
    return _pair_runs_with_opponents(df, team_index, runs, "torso_fall")


def _pair_runs_with_opponents(df: pd.DataFrame, team_index: dict, runs: list[dict], trigger: str) -> list[dict]:
    """Shared by both pose-based triggers: for each "player is down" run,
    finds any nearby opposing-team player and turns the pair into a
    candidate. See `find_pose_collapse_candidates`'s docstring for why
    proximity uses a narrow-then-broad window and why only the OTHER
    player's speed is checked against MAX_CLOSING_SPEED_MPS."""
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
            if other["speed_mps"] > MAX_CLOSING_SPEED_MPS:
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
                "closing_speed_mps": other["speed_mps"],
                "trigger": trigger,
            })
    return sorted(candidates, key=lambda c: c["time_s"])


def find_keypoint_contact_candidates(player_time_df: pd.DataFrame) -> list[dict]:
    """Promotes real joint-to-joint contact (src/events/pose_signals.
    contact_type_events) into first-class foul candidates -- see
    KEYPOINT_CONTACT_TRIGGER_TYPES's comment for why this exists.

    Deliberately does NOT apply MAX_CLOSING_SPEED_MPS, unlike every trigger
    above: that cap exists to reject candidates whose only evidence of
    contact is an implausible closing speed (a tracking-ID-switch artifact
    masquerading as a collision). Here, the evidence of contact is direct
    joint-to-joint proximity -- a strictly stronger signal than box
    geometry or speed -- so a genuine contact confirmed at the keypoint
    level should not be discarded just because the same violent event also
    corrupted the derived speed estimate. That was confirmed to be exactly
    what already blocks the pose-collapse and torso-fall triggers on
    Chelsea-Burnley (the dev log, 2026-07-16); repeating the same cap here
    would reproduce the identical failure for the one trigger meant to
    route around it. closing_speed_mps is still computed and carried
    through (for Module C's severity reasoning), it just isn't a gate."""
    from src.events import pose_signals  # local import: pose_signals imports team-lookup helpers from this module

    contacts = pose_signals.contact_type_events(player_time_df)
    candidates = []
    last_flagged_t: dict = {}
    for c in contacts:
        if c["contact_type"] not in KEYPOINT_CONTACT_TRIGGER_TYPES:
            continue
        pair_key = tuple(sorted((c["track_id_a"], c["track_id_b"])))
        if c["time_s"] - last_flagged_t.get(pair_key, -999) < MIN_GAP_S:
            continue
        last_flagged_t[pair_key] = c["time_s"]
        candidates.append({
            "time_s": c["time_s"],
            "track_id_a": c["track_id_a"], "team_a": c["team_a"],
            "track_id_b": c["track_id_b"], "team_b": c["team_b"],
            "location": c["location"],
            "closing_speed_mps": c["closing_speed_mps"],
            "trigger": "keypoint_contact",
        })
    return sorted(candidates, key=lambda c: c["time_s"])


def find_contact_candidates(player_time_df: pd.DataFrame) -> list[dict]:
    """Union of four independent triggers: the original distance+speed
    gate (unchanged), the box-aspect-ratio pose-collapse gate (see
    `find_pose_collapse_candidates`), the torso-angle fall gate (see
    `find_torso_fall_candidates`), and the keypoint-contact gate (see
    `find_keypoint_contact_candidates`). A later trigger's hit for a pair
    already flagged by an earlier one around the same time is merged into
    that same candidate (its name appended to the "triggers" list) rather
    than appended as a separate duplicate event for what is very likely
    the same real contact."""
    candidates = _distance_speed_candidates(player_time_df)
    for c in candidates:
        c["triggers"] = [c.pop("trigger")]

    def _merge_in(new_candidates: list[dict]) -> None:
        for pc in new_candidates:
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

    _merge_in(find_pose_collapse_candidates(player_time_df))
    _merge_in(find_torso_fall_candidates(player_time_df))
    _merge_in(find_keypoint_contact_candidates(player_time_df))

    return sorted(candidates, key=lambda c: c["time_s"])
