"""Downloads and trims a real clip around a genuine SoccerNet "Red card"
action-spotting label (Labels-v2.json, see notebooks/find_card_events_v2.py)
-- unlike this project's existing card_*.mp4 clips, which were picked before
any scene-cut cross-referencing existed and used the older 3-class
Labels.json schema only to confirm a card happened, not to independently
verify clip cleanliness. Hull City vs Arsenal is a fresh match, not
overlapping any existing clip in this project's set.

Usage:
    python -m notebooks.download_card_v2_clip
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

JOB = {
    "slug": "card_hull_arsenal",
    "game": "england_epl/2016-2017/2016-09-17 - 17-00 Hull City 1 - 4 Arsenal",
    "half": 1, "card_time_s": 39 * 60 + 20, "team": "home", "label": "Red card",
}


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
    start_frame = int((job["card_time_s"] - WINDOW_BEFORE_S) * fps)
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

    shutil.rmtree(os.path.join(DEST_DIR, job["game"].split("/")[0]))
    return out_path


if __name__ == "__main__":
    print(f"Downloading/trimming {JOB['slug']} ({JOB['game']}, half {JOB['half']}, "
          f"{JOB['label']} at {JOB['card_time_s']}s)...")
    path = download_and_trim(JOB)
    print(f"  -> wrote {path}")
