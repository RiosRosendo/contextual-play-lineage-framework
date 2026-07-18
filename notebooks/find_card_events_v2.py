"""Searches SoccerNet's freely-downloadable Labels-v2.json (no password
needed) for its "Yellow card" / "Red card" action-spotting labels --
same method as notebooks/find_foul_events_v2.py, applied to cards instead
of fouls, to find a cleaner card test clip than the existing card_*.mp4
set (all of which used the older 3-class Labels.json schema, no
cross-referencing against scene-cut density at the time they were picked).

Filters for isolated cards (no other card within ISOLATION_WINDOW_S in the
same half) and away from kickoff/half-end.

Usage:
    python -m notebooks.find_card_events_v2
"""
from __future__ import annotations

import json
from pathlib import Path

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames

LABELS_DIR = "data/raw/soccernet_labels_v2"
CARD_LABELS = ("Yellow card", "Red card", "Yellow->red card")
ISOLATION_WINDOW_S = 20
MIN_HALF_CLOCK_S = 120
MAX_HALF_CLOCK_S = 2500


def _parse_game_time(game_time: str) -> tuple[int, int]:
    half_str, clock = game_time.split(" - ")
    minutes, seconds = clock.split(":")
    return int(half_str), int(minutes) * 60 + int(seconds)


def download_labels_batch(n_games: int = 60) -> list[str]:
    downloader = SoccerNetDownloader(LocalDirectory=LABELS_DIR)
    games = getListGames("train", task="spotting")[:n_games]
    for game in games:
        downloader.downloadGame(game=game, files=["Labels-v2.json"], spl="train", verbose=False)
    return games


def find_isolated_cards(games: list[str]) -> list[dict]:
    candidates = []
    for game in games:
        path = Path(LABELS_DIR) / game / "Labels-v2.json"
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
                continue
            candidates.append({
                "game": game, "half": card["half"], "sec": card["sec"],
                "gameTime": card["gameTime"], "team": card.get("team"),
                "label": card["label"], "visibility": card.get("visibility"),
            })
    return candidates


if __name__ == "__main__":
    games = download_labels_batch()
    print(f"Searched {len(games)} matches (Labels-v2.json).")
    candidates = find_isolated_cards(games)
    print(f"Found {len(candidates)} isolated card incident(s):\n")
    for c in candidates:
        print(f"{c['game']}")
        print(f"  {c['gameTime']}  {c['label']}  team={c['team']}  visibility={c['visibility']}")
