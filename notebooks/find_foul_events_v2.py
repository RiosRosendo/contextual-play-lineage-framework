"""Searches SoccerNet's freely-downloadable Labels-v2.json (no password
needed, unlike the video) for its actual "Foul" action-spotting label --
unlike Labels.json (the 3-class card/goal/sub schema this project has used
so far as a foul proxy, see notebooks/find_foul_before_goal.py), Labels-v2
uses the full 17-class action schema and has a real "Foul" annotation.

Filters for isolated fouls (no other Foul within ISOLATION_WINDOW_S in the
same half, so a candidate isn't confounded by a second nearby incident) and
away from kickoff/half-end (needs a clean before/after window for the
5+ second pre-contact feature extraction Layer 3 uses).

Usage:
    python -m notebooks.find_foul_events_v2
"""
from __future__ import annotations

import json
from pathlib import Path

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames

LABELS_DIR = "data/raw/soccernet_labels_v2"
ISOLATION_WINDOW_S = 15  # no other Foul within this many seconds, same half
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


def find_isolated_fouls(games: list[str]) -> list[dict]:
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
        fouls = [e for e in events if e["label"] == "Foul"]

        for foul in fouls:
            if not (MIN_HALF_CLOCK_S <= foul["sec"] <= MAX_HALF_CLOCK_S):
                continue
            others_nearby = [
                f for f in fouls
                if f is not foul and f["half"] == foul["half"]
                and abs(f["sec"] - foul["sec"]) <= ISOLATION_WINDOW_S
            ]
            if others_nearby:
                continue
            candidates.append({
                "game": game, "half": foul["half"], "sec": foul["sec"],
                "gameTime": foul["gameTime"], "team": foul.get("team"),
                "visibility": foul.get("visibility"),
            })
    return candidates


if __name__ == "__main__":
    games = download_labels_batch()
    print(f"Searched {len(games)} matches (Labels-v2.json, freely downloadable, no password).")
    candidates = find_isolated_fouls(games)
    print(f"Found {len(candidates)} isolated Foul incident(s):\n")
    for c in candidates:
        print(f"{c['game']}")
        print(f"  {c['gameTime']}  team={c['team']}  visibility={c['visibility']}")
