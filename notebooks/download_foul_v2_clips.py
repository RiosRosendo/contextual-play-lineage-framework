"""Downloads and trims real clips around genuine SoccerNet "Foul" action-
spotting labels (Labels-v2.json, see notebooks/find_foul_events_v2.py) --
unlike every other real clip in this project so far, which used a card as
a proxy for "a foul happened nearby" since only the 3-class Labels.json
schema was available. These two candidates were picked because their
match wasn't already used elsewhere in this project (avoiding overlap with
the existing card-clip set) and their Foul annotation is `visibility:
"visible"` (not a replay-only/off-camera event).

Same download discipline as notebooks/download_soccernet_card_clips.py:
only the needed half is downloaded, and the full (~1GB) half is deleted
immediately after trimming.

Usage:
    python -m notebooks.download_foul_v2_clips
"""
from __future__ import annotations

import os
import shutil

import cv2
from dotenv import load_dotenv

DEST_DIR = "data/raw/soccernet"
SPLIT = "train"
WINDOW_BEFORE_S = 10
WINDOW_AFTER_S = 10

FOUL_JOBS = [
    {
        "slug": "foul_leicester_mancity",
        "game": "england_epl/2016-2017/2016-12-10 - 20-30 Leicester 4 - 2 Manchester City",
        "half": 1, "foul_time_s": 18 * 60 + 26, "team": "home",
    },
    {
        "slug": "foul_arsenal_anderlecht",
        "game": "europe_uefa-champions-league/2014-2015/2014-11-04 - 22-45 Arsenal 3 - 3 Anderlecht",
        "half": 1, "foul_time_s": 19 * 60 + 48, "team": "away",
    },
]


def download_and_trim(job: dict) -> str:
    load_dotenv()
    password = os.environ.get("SOCCERNET_PASSWORD")
    if not password:
        raise RuntimeError("SOCCERNET_PASSWORD not set. Add it to a local .env file.")

    from SoccerNet.Downloader import SoccerNetDownloader

    half_file = f"{job['half']}_720p.mkv"
    out_path = f"{DEST_DIR}/{job['slug']}.mp4"

    downloader = SoccerNetDownloader(LocalDirectory=DEST_DIR)
    downloader.password = password
    downloader.downloadGame(game=job["game"], files=[half_file], spl=SPLIT, verbose=True)
    full_path = os.path.join(DEST_DIR, job["game"], half_file)

    cap = cv2.VideoCapture(full_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int((job["foul_time_s"] - WINDOW_BEFORE_S) * fps)
    n_frames = int((WINDOW_BEFORE_S + WINDOW_AFTER_S) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
    out.release()
    cap.release()

    shutil.rmtree(os.path.join(DEST_DIR, job["game"].split("/")[0]))  # drop the ~1GB source
    return out_path


if __name__ == "__main__":
    for job in FOUL_JOBS:
        print(f"Downloading/trimming {job['slug']} ({job['game']}, half {job['half']}, "
              f"foul at {job['foul_time_s']}s)...")
        path = download_and_trim(job)
        print(f"  -> wrote {path}")
