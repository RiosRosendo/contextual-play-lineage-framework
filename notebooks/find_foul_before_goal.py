"""Searches SoccerNet's freely-downloadable Labels.json (no password
needed, unlike the video) across many matches for a card shortly before a
goal in the same half -- the best available proxy for "an unflagged foul
immediately preceding a goal" given this label schema only has
card/goal/substitution events, no explicit "foul" label. A card within a
short window of a goal is a reasonable (not perfect) signal that a foul
happened in the buildup.

Usage:
    python -m notebooks.find_foul_before_goal
"""
from __future__ import annotations

import json
import re

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames

LABELS_DIR = "data/raw/soccernet_labels"
CARD_TO_GOAL_WINDOW_S = 90  # how close a card must be before a goal to count as a candidate


def _parse_game_time(game_time: str) -> tuple[int, int]:
    """'1 - 20:41' -> (half=1, seconds=1241)"""
    half_str, clock = game_time.split(" - ")
    minutes, seconds = clock.split(":")
    return int(half_str), int(minutes) * 60 + int(seconds)


def download_labels_batch(n_games: int = 80) -> list[str]:
    downloader = SoccerNetDownloader(LocalDirectory=LABELS_DIR)
    games = getListGames("train", task="spotting")[:n_games]
    for game in games:
        downloader.downloadGame(game=game, files=["Labels.json"], spl="train", verbose=False)
    return games


def find_candidates(games: list[str]) -> list[dict]:
    candidates = []
    for game in games:
        path = f"{LABELS_DIR}/{game}/Labels.json"
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            continue

        events = [
            {**a, "half": _parse_game_time(a["gameTime"])[0], "sec": _parse_game_time(a["gameTime"])[1]}
            for a in data["annotations"]
        ]
        goals = [e for e in events if e["label"] == "soccer-ball"]
        cards = [e for e in events if e["label"] in ("y-card", "r-card", "y2r-card")]

        for goal in goals:
            nearby = [
                c for c in cards
                if c["half"] == goal["half"] and 0 <= goal["sec"] - c["sec"] <= CARD_TO_GOAL_WINDOW_S
            ]
            if nearby:
                candidates.append({
                    "game": game, "goal": goal, "cards_before": nearby,
                    "gap_s": goal["sec"] - max(c["sec"] for c in nearby),
                })
    return sorted(candidates, key=lambda c: c["gap_s"])


if __name__ == "__main__":
    games = download_labels_batch()
    print(f"Searched {len(games)} matches.")
    candidates = find_candidates(games)
    print(f"Found {len(candidates)} candidate(s) with a card within {CARD_TO_GOAL_WINDOW_S}s before a goal:\n")
    for c in candidates:
        print(f"{c['game']}")
        print(f"  goal: {c['goal']['gameTime']} (team {c['goal']['team']})")
        print(f"  card(s) before: {[(x['gameTime'], x['label'], x['team']) for x in c['cards_before']]}")
        print(f"  gap: {c['gap_s']}s\n")
