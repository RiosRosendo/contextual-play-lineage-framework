"""Keypoint-derived event signals over Layer 1's full-body skeleton output
(the kp_<joint>_x/_y/_c columns produced by the dual-pass pose extraction in
src/perception/pipeline.py). Three capability families, all deliberately
simple threshold heuristics in the skeleton-first spirit of the project
spec, not trained models:

1. Contact-type identification between two opposing players' body parts --
   hand-to-face, elbow-to-body, shirt-pull (sustained wrist-to-torso),
   leg contact. Richer context than "two boxes were near each other" for
   downstream foul reasoning (src/assistant/explain.py can consume the
   `contact_types` annotation on foul events without further plumbing).
2. Handball candidates: a wrist keypoint near the tracked ball. Honest
   limitations, disclosed up front: goalkeepers (who may legally handle
   the ball) are not distinguishable from outfield players yet, and
   throw-ins are legal handling too -- so these are review CANDIDATES with
   expected false positives from both, not verdicts.
3. Pose analytics for the state store / player-performance work: jump
   height during aerial actions (hip rise relative to the player's own
   box height -- screen-space, so camera motion contaminates it; disclosed,
   not hidden) and sprint stride cadence (ankle-separation oscillation).

All distances here are pixel-space, scaled by the players' own box sizes
(same reasoning as the pose-collapse trigger in contact_candidates.py:
calibration reliability is a documented weak point, so nothing here
depends on it). All keypoint reads require per-joint confidence >=
KEYPOINT_CONF_MIN; a joint below that simply doesn't participate.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.events.foul_detector.contact_candidates import (
    _build_known_team_index, _nearest_known_team,
)

KEYPOINT_CONF_MIN = 0.3

# Body-region groups by kp_<name> column stem (COCO joints; the wrist is the
# closest available proxy for the hand -- COCO has no hand keypoint).
HEAD = ("nose", "l_eye", "r_eye", "l_ear", "r_ear")
TORSO = ("l_shoulder", "r_shoulder", "l_hip", "r_hip")
ELBOWS = ("l_elbow", "r_elbow")
HANDS = ("l_wrist", "r_wrist")
LEGS = ("l_knee", "r_knee", "l_ankle", "r_ankle")

# (attacker region, victim region, contact type, min consecutive frames).
# Single-frame for strikes (contact is instantaneous); shirt-pull needs to
# be SUSTAINED -- a grip held over time is exactly what distinguishes it
# from a hand incidentally brushing past a torso.
CONTACT_RULES = (
    (HANDS, HEAD, "hand_to_face", 1),
    (ELBOWS, TORSO, "elbow_to_body", 1),
    (HANDS, TORSO, "shirt_pull", 5),
    (LEGS, LEGS, "leg_contact", 1),
)

# Contact if the closest attacker-joint-to-victim-joint distance is under
# this fraction of the two players' mean box height (box height ~ player
# height ~1.8m, so 0.2 ~ 0.35m reach -- generous enough for motion blur,
# tight enough that a neighbor a meter away doesn't trigger).
CONTACT_FRAC_OF_HEIGHT = 0.2
CONTACT_EVENT_GAP_S = 1.0  # debounce repeated hits for the same pair+type

# Handball: wrist within this fraction of the player's box height of the
# ball's box center.
HANDBALL_FRAC_OF_HEIGHT = 0.15
HANDBALL_EVENT_GAP_S = 1.0

# Jump detection: hip midpoint must rise at least this fraction of the
# player's own baseline box height above its rolling baseline, for at
# least 2 consecutive frames.
JUMP_MIN_RISE_FRAC = 0.25
JUMP_MIN_FRAMES = 2
JUMP_BASELINE_WINDOW_S = 1.0

# Sprint cadence: only measured while a track sustains at least this speed
# (speed_mps comes from calibrated positions, which are noisy -- this is a
# coarse gate, not a precise one), over windows of at least this length.
SPRINT_MIN_SPEED_MPS = 5.0
SPRINT_MIN_WINDOW_S = 1.5


def _region_points(row: pd.Series, region: tuple) -> list[tuple[float, float]]:
    pts = []
    for name in region:
        c = row.get(f"kp_{name}_c")
        if c is not None and not pd.isna(c) and c >= KEYPOINT_CONF_MIN:
            pts.append((row[f"kp_{name}_x"], row[f"kp_{name}_y"]))
    return pts


def _min_region_distance(row_a: pd.Series, region_a: tuple,
                          row_b: pd.Series, region_b: tuple) -> float | None:
    pts_a, pts_b = _region_points(row_a, region_a), _region_points(row_b, region_b)
    if not pts_a or not pts_b:
        return None
    return min(math.hypot(ax - bx, ay - by) for ax, ay in pts_a for bx, by in pts_b)


def _box_height(row: pd.Series) -> float:
    return float(row["box_y2"] - row["box_y1"])


def contact_type_events(player_time_df: pd.DataFrame) -> list[dict]:
    """Scans every frame for opposing-player pairs whose keypoints satisfy a
    CONTACT_RULES entry. Directional: (a, b) means a's attacker-region
    joint touched b's victim-region -- both directions are checked."""
    players = player_time_df[player_time_df["cls"] == "player"]
    if players.empty or "kp_nose_x" not in players.columns:
        return []
    team_index = _build_known_team_index(players)

    # (pair_key, type) -> list of consecutive frame hits, for the sustained rules
    streaks: dict[tuple, dict] = {}
    last_event_t: dict[tuple, float] = {}
    events = []

    for frame, group in players.groupby("frame"):
        rows = group.to_dict("records")
        t = rows[0]["time_s"]
        for i in range(len(rows)):
            for j in range(len(rows)):
                if i == j:
                    continue
                a, b = pd.Series(rows[i]), pd.Series(rows[j])
                team_a = _nearest_known_team(team_index, a["track_id"], t)
                team_b = _nearest_known_team(team_index, b["track_id"], t)
                if team_a is None or team_b is None or team_a == team_b:
                    continue
                mean_h = (_box_height(a) + _box_height(b)) / 2
                if mean_h <= 0:
                    continue
                thresh = CONTACT_FRAC_OF_HEIGHT * mean_h

                for region_a, region_b, ctype, min_frames in CONTACT_RULES:
                    dist = _min_region_distance(a, region_a, b, region_b)
                    key = (a["track_id"], b["track_id"], ctype)
                    if dist is None or dist > thresh:
                        streaks.pop(key, None)
                        continue
                    streak = streaks.setdefault(key, {"n": 0, "start_t": t, "min_dist": dist})
                    streak["n"] += 1
                    streak["min_dist"] = min(streak["min_dist"], dist)
                    if streak["n"] < min_frames:
                        continue
                    if t - last_event_t.get(key, -999) < CONTACT_EVENT_GAP_S:
                        continue
                    last_event_t[key] = t
                    events.append({
                        "type": "contact", "contact_type": ctype,
                        "time_s": streak["start_t"],
                        "track_id_a": a["track_id"], "team_a": team_a,
                        "track_id_b": b["track_id"], "team_b": team_b,
                        "min_dist_px": streak["min_dist"],
                        "dist_frac_of_height": streak["min_dist"] / mean_h,
                        # Carried through so contact_candidates.py can promote
                        # a contact event directly into a foul candidate
                        # (find_keypoint_contact_candidates) without a
                        # separate position/speed lookup.
                        "location": ((a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2),
                        "closing_speed_mps": float(a.get("speed_mps") or 0.0) + float(b.get("speed_mps") or 0.0),
                    })
    return sorted(events, key=lambda e: e["time_s"])


def handball_events(player_time_df: pd.DataFrame) -> list[dict]:
    """Wrist-to-ball proximity candidates. See module docstring for the two
    disclosed false-positive sources (goalkeepers, throw-ins)."""
    players = player_time_df[player_time_df["cls"] == "player"]
    balls = player_time_df[player_time_df["cls"] == "ball"]
    if players.empty or balls.empty or "kp_l_wrist_x" not in players.columns:
        return []
    team_index = _build_known_team_index(players)

    ball_by_frame = {}
    for _, brow in balls.iterrows():
        ball_by_frame[brow["frame"]] = (
            (brow["box_x1"] + brow["box_x2"]) / 2, (brow["box_y1"] + brow["box_y2"]) / 2,
        )

    last_event_t: dict = {}
    events = []
    for _, row in players.iterrows():
        ball = ball_by_frame.get(row["frame"])
        if ball is None:
            continue
        wrists = _region_points(row, HANDS)
        if not wrists:
            continue
        h = _box_height(row)
        if h <= 0:
            continue
        dist = min(math.hypot(wx - ball[0], wy - ball[1]) for wx, wy in wrists)
        if dist > HANDBALL_FRAC_OF_HEIGHT * h:
            continue
        t = row["time_s"]
        if t - last_event_t.get(row["track_id"], -999) < HANDBALL_EVENT_GAP_S:
            continue
        last_event_t[row["track_id"]] = t
        events.append({
            "type": "handball_candidate", "time_s": t,
            "track_id": row["track_id"],
            "team": _nearest_known_team(team_index, row["track_id"], t),
            "wrist_ball_dist_px": dist, "dist_frac_of_height": dist / h,
        })
    return sorted(events, key=lambda e: e["time_s"])


def annotate_foul_contact_types(foul_events: list[dict], contact_events: list[dict],
                                 window_s: float = 1.0) -> None:
    """Attaches a `contact_types` list to each foul event: the distinct
    keypoint-level contact types observed between the same two tracks
    within +/-window_s of the foul. In-place; events with no matching
    contact get an empty list (an honest "boxes were close but no
    joint-level contact was resolved"), never a guess."""
    for foul in foul_events:
        pair = {foul.get("track_id_a"), foul.get("track_id_b")}
        types = {
            c["contact_type"] for c in contact_events
            if abs(c["time_s"] - foul["time_s"]) <= window_s
            and {c["track_id_a"], c["track_id_b"]} == pair
        }
        foul["contact_types"] = sorted(types)


def jump_events(player_time_df: pd.DataFrame) -> list[dict]:
    """Per track, detects aerial actions: the hip midpoint rising at least
    JUMP_MIN_RISE_FRAC of the player's own baseline box height above its
    rolling-baseline position for JUMP_MIN_FRAMES+ consecutive frames.
    `jump_height_m` converts the rise using the player's own box height as
    a ~1.8m yardstick -- screen-space, uncalibrated, contaminated by camera
    motion; treat as indicative, not measured."""
    players = player_time_df[player_time_df["cls"] == "player"]
    if players.empty or "kp_l_hip_y" not in players.columns:
        return []

    dt = players["time_s"].diff().median()
    if not dt or dt <= 0:
        return []
    window = max(3, round(JUMP_BASELINE_WINDOW_S / dt))

    events = []
    for track_id, g in players.sort_values("frame").groupby("track_id"):
        g = g.reset_index(drop=True)
        hip_y = g[["kp_l_hip_y", "kp_r_hip_y"]].mean(axis=1)
        conf_ok = (g["kp_l_hip_c"] >= KEYPOINT_CONF_MIN) | (g["kp_r_hip_c"] >= KEYPOINT_CONF_MIN)
        hip_y = hip_y.where(conf_ok)
        box_h = g["box_y2"] - g["box_y1"]
        baseline_y = hip_y.rolling(window=window, min_periods=3).median().shift(1)
        baseline_h = box_h.rolling(window=window, min_periods=3).median().shift(1)
        rise_frac = ((baseline_y - hip_y) / baseline_h.where(baseline_h > 0))

        in_jump = (rise_frac > JUMP_MIN_RISE_FRAC).fillna(False)
        start = None
        for i, flag in enumerate(in_jump):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                if i - start >= JUMP_MIN_FRAMES:
                    seg = rise_frac.iloc[start:i]
                    peak = float(seg.max())
                    events.append({
                        "type": "jump", "time_s": float(g["time_s"].iloc[start]),
                        "track_id": track_id, "peak_rise_frac": peak,
                        "jump_height_m": peak * 1.8,
                    })
                start = None
    return sorted(events, key=lambda e: e["time_s"])


def sprint_cadence(player_time_df: pd.DataFrame) -> list[dict]:
    """Stride cadence (steps/second) for sustained fast-running windows,
    from the oscillation of the ankle-separation signal |l_ankle_x -
    r_ankle_x| (one separation peak per step). Returns one summary row per
    qualifying (track, window)."""
    players = player_time_df[player_time_df["cls"] == "player"]
    if players.empty or "kp_l_ankle_x" not in players.columns:
        return []

    results = []
    for track_id, g in players.sort_values("frame").groupby("track_id"):
        g = g.reset_index(drop=True)
        fast = (g["speed_mps"] >= SPRINT_MIN_SPEED_MPS) if "speed_mps" in g else pd.Series(False, index=g.index)
        conf_ok = (g["kp_l_ankle_c"] >= KEYPOINT_CONF_MIN) & (g["kp_r_ankle_c"] >= KEYPOINT_CONF_MIN)
        usable = (fast & conf_ok).to_numpy()

        start = None
        for i in range(len(g) + 1):
            flag = usable[i] if i < len(g) else False
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                seg = g.iloc[start:i]
                duration = float(seg["time_s"].iloc[-1] - seg["time_s"].iloc[0])
                if duration >= SPRINT_MIN_WINDOW_S:
                    sep = (seg[f"kp_l_ankle_x"] - seg[f"kp_r_ankle_x"]).abs().to_numpy()
                    peaks = sum(
                        1 for k in range(1, len(sep) - 1)
                        if sep[k] > sep[k - 1] and sep[k] >= sep[k + 1]
                    )
                    results.append({
                        "type": "sprint_window", "track_id": track_id,
                        "time_s": float(seg["time_s"].iloc[0]), "duration_s": duration,
                        "mean_speed_mps": float(seg["speed_mps"].mean()),
                        "steps_per_s": peaks / duration if duration > 0 else 0.0,
                    })
                start = None
    return sorted(results, key=lambda e: e["time_s"])
