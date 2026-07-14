"""Finds candidate foul moments: frames where two opposing players are
within a contact distance and closing fast. This drives what windows get
fed to the two-branch classifier -- CLAUDE.md section 4 (Layer 3) says the
detector looks at the 5+ seconds before contact, so the "event" here is the
contact instant, and features.py extracts the pre-contact window around it.
"""
from __future__ import annotations

import pandas as pd

CONTACT_DIST_M = 1.5
MIN_CLOSING_SPEED_MPS = 3.0
MIN_GAP_S = 1.0  # avoid re-flagging the same contact on consecutive frames


def find_contact_candidates(player_time_df: pd.DataFrame) -> list[dict]:
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
                if closing_speed < MIN_CLOSING_SPEED_MPS:
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
                })
    return candidates
