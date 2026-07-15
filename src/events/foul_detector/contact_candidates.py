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
POSE_COLLAPSE_MIN_FRAMES = 3  # must stay collapsed this many consecutive frames -- not a single noisy frame
POSE_OVERLAP_TOLERANCE_S = 1.0  # two tracks' collapse windows count as "the same moment" within this gap
POSE_PROXIMITY_BOX_DIAGONALS = 2.0  # "nearby" boxes, scaled by box size so it holds at any camera distance
POSE_MERGE_WINDOW_S = 2.0  # merge a pose-collapse hit into an existing distance/speed hit for the same pair


def _distance_speed_candidates(player_time_df: pd.DataFrame) -> list[dict]:
    players = player_time_df[player_time_df["cls"] == "player"]
    candidates = []
    last_flagged_t = {}

    for frame, group in players.groupby("frame"):
        rows = group.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if a["team"] == b["team"] or a["team"] is None or b["team"] is None:
                    continue
                dist = ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5
                if dist > CONTACT_DIST_M:
                    continue
                closing_speed = a.get("speed_mps", 0.0) + b.get("speed_mps", 0.0)
                if closing_speed < MIN_CLOSING_SPEED_MPS or closing_speed > MAX_CLOSING_SPEED_MPS:
                    continue
                pair_key = tuple(sorted((a["track_id"], b["track_id"])))
                t = a["time_s"]
                if t - last_flagged_t.get(pair_key, -999) < MIN_GAP_S:
                    continue
                last_flagged_t[pair_key] = t
                candidates.append({
                    "time_s": t, "frame": frame,
                    "track_id_a": a["track_id"], "team_a": a["team"],
                    "track_id_b": b["track_id"], "team_b": b["team"],
                    "location": ((a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2),
                    "closing_speed_mps": closing_speed,
                    "trigger": "distance_speed",
                })
    return candidates


def _collapse_runs(player_time_df: pd.DataFrame) -> list[dict]:
    """Per track, finds sustained aspect-ratio-collapse runs (a box going
    tall-and-narrow -> short-and-wide for several consecutive frames --
    falling, being tackled, ending up tangled on the ground)."""
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
        collapsed = (aspect < POSE_COLLAPSE_RATIO * baseline).fillna(False)

        start = None
        for i, is_collapsed in enumerate(collapsed):
            if is_collapsed and start is None:
                start = i
            elif not is_collapsed and start is not None:
                if i - start >= POSE_COLLAPSE_MIN_FRAMES:
                    runs.append(_run_dict(track_id, g, start, i - 1))
                start = None
        if start is not None and len(g) - start >= POSE_COLLAPSE_MIN_FRAMES:
            runs.append(_run_dict(track_id, g, start, len(g) - 1))
    return runs


def _run_dict(track_id, g: pd.DataFrame, i0: int, i1: int) -> dict:
    seg = g.iloc[i0:i1 + 1]
    diag = ((seg["box_x2"] - seg["box_x1"]) ** 2 + (seg["box_y2"] - seg["box_y1"]) ** 2) ** 0.5
    return {
        "track_id": track_id, "team": seg["team"].iloc[0],
        "start_time_s": float(seg["time_s"].iloc[0]), "end_time_s": float(seg["time_s"].iloc[-1]),
        "cx": float(((seg["box_x1"] + seg["box_x2"]) / 2).mean()),
        "cy": float(((seg["box_y1"] + seg["box_y2"]) / 2).mean()),
        "diag": float(diag.mean()),
        "x": float(seg["x"].mean()), "y": float(seg["y"].mean()),
        "speed_mps": float(seg["speed_mps"].mean()) if "speed_mps" in seg else 0.0,
    }


def find_pose_collapse_candidates(player_time_df: pd.DataFrame) -> list[dict]:
    """Flags a candidate whenever two opposing players' boxes BOTH show a
    sustained aspect-ratio collapse within the same short window and are
    "nearby" in pixel space (scaled by box size, NOT the calibrated meter
    distance -- see module docstring for why) -- independent of whether
    `_distance_speed_candidates` would also have fired for the same
    moment."""
    runs = _collapse_runs(player_time_df)
    candidates = []
    seen_pairs = set()
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = runs[i], runs[j]
            if a["track_id"] == b["track_id"] or a["team"] == b["team"] or a["team"] is None or b["team"] is None:
                continue
            gap = max(a["start_time_s"], b["start_time_s"]) - min(a["end_time_s"], b["end_time_s"])
            if gap > POSE_OVERLAP_TOLERANCE_S:
                continue
            dist_px = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
            avg_diag = (a["diag"] + b["diag"]) / 2
            if avg_diag <= 0 or dist_px > POSE_PROXIMITY_BOX_DIAGONALS * avg_diag:
                continue
            pair_key = tuple(sorted((a["track_id"], b["track_id"])))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            t = min(a["start_time_s"], b["start_time_s"])
            candidates.append({
                "time_s": t,
                "track_id_a": a["track_id"], "team_a": a["team"],
                "track_id_b": b["track_id"], "team_b": b["team"],
                "location": ((a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2),
                "closing_speed_mps": a["speed_mps"] + b["speed_mps"],
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
