"""Searches the already-downloaded SoccerNet Labels.json cache (from
notebooks/find_foul_before_goal.py's 80-match batch) for "clean" single-card
incidents: a card with no other card event nearby in the same half, in
either team. Used to build a small, diverse real-card test set for the
foul-verdict-audit capability (src/assistant/explain.py), which needs more
than the one already-used Sunderland-Liverpool double-yellow-card moment to
report an aggregate accuracy table instead of a single anecdote.

Deliberately excludes double/clustered card moments (like the
Sunderland-Liverpool one already used) since those make it harder to
attribute which detected contact candidate corresponds to which card.

Usage:
    python -m notebooks.find_single_card_incidents
"""
from __future__ import annotations

import json
from pathlib import Path

LABELS_DIR = "data/raw/soccernet_labels"
CARD_LABELS = ("y-card", "r-card", "y2r-card")
ISOLATION_WINDOW_S = 20  # no other card within this many seconds, either team
MIN_HALF_CLOCK_S = 120  # avoid cards too close to kickoff (need a clean before/after window)
MAX_HALF_CLOCK_S = 2500  # avoid cards too close to a half's end


def _parse_game_time(game_time: str) -> tuple[int, int]:
    half_str, clock = game_time.split(" - ")
    minutes, seconds = clock.split(":")
    return int(half_str), int(minutes) * 60 + int(seconds)


def find_isolated_cards() -> list[dict]:
    candidates = []
    for path in Path(LABELS_DIR).glob("*/*/*/Labels.json"):
        game = str(path.parent.relative_to(LABELS_DIR))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        events = [
            {**a, "half": _parse_game_time(a["gameTime"])[0], "sec": _parse_game_time(a["gameTime"])[1]}
            for a in data["annotations"]
        ]
        cards = [e for e in events if e["label"] in CARD_LABELS]

        for card in cards:
            if not (MIN_HALF_CLOCK_S <= card["sec"] <= MAX_HALF_CLOCK_S):
                continue
            others_nearby = [
                c for c in cards
                if c is not card and c["half"] == card["half"]
                and abs(c["sec"] - card["sec"]) <= ISOLATION_WINDOW_S
            ]
            if others_nearby:
                continue  # not isolated -- skip, same reasoning as the double-yellow case
            candidates.append({
                "game": game, "half": card["half"], "sec": card["sec"],
                "gameTime": card["gameTime"], "label": card["label"], "team": card["team"],
            })
    return candidates


if __name__ == "__main__":
    candidates = find_isolated_cards()
    print(f"Found {len(candidates)} isolated single-card incident(s) across the cached matches.\n")
    seen_games = set()
    for c in sorted(candidates, key=lambda c: c["game"]):
        marker = "" if c["game"] not in seen_games else "  (extra card, same match already listed)"
        seen_games.add(c["game"])
        print(f"{c['game']}")
        print(f"  {c['gameTime']}  {c['label']}  team={c['team']}{marker}")
