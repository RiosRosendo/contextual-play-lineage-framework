"""Downloads a small amount of real SoccerNet broadcast footage (now that
NDA access is approved) for the real-footage diagnostic pass -- just
enough for a short validation clip, not the full 500+ game archive.

Requires SOCCERNET_PASSWORD in a local .env file (gitignored, never
committed -- see .env.example if present, or just add the line yourself).

The full downloaded half (~1GB) is deleted right after trimming -- only the
short clip is kept, since that's all the diagnostic pass needs.

Usage:
    python -m notebooks.download_soccernet_clip
"""
from __future__ import annotations

import os
import shutil

import cv2
from dotenv import load_dotenv

GAME = "spain_laliga/2016-2017/2016-08-28 - 21-15 Ath Bilbao 0 - 1 Barcelona"
SPLIT = "train"
DEST_DIR = "data/raw/soccernet"
CLIP_START_FRAME = 8000  # ~05:20 in-game clock; well past kickoff, into open play
CLIP_N_FRAMES = 500  # 20s at 25fps
CLIP_OUT_PATH = "data/raw/soccernet/clip20s.mp4"


def download_full_half() -> str:
    load_dotenv()
    password = os.environ.get("SOCCERNET_PASSWORD")
    if not password:
        raise RuntimeError(
            "SOCCERNET_PASSWORD not set. Add it to a local .env file "
            "(SOCCERNET_PASSWORD=...); .env is gitignored."
        )

    from SoccerNet.Downloader import SoccerNetDownloader

    downloader = SoccerNetDownloader(LocalDirectory=DEST_DIR)
    downloader.password = password
    downloader.downloadGame(game=GAME, files=["1_720p.mkv"], spl=SPLIT, verbose=True)
    return os.path.join(DEST_DIR, GAME, "1_720p.mkv")


def trim_clip(source_path: str) -> str:
    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, CLIP_START_FRAME)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(CLIP_OUT_PATH, fourcc, fps, (w, h))
    for _ in range(CLIP_N_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
    out.release()
    cap.release()
    return CLIP_OUT_PATH


def download_and_trim() -> str:
    full_path = download_full_half()
    clip_path = trim_clip(full_path)
    shutil.rmtree(os.path.join(DEST_DIR, GAME.split("/")[0]))  # drop the ~1GB source
    return clip_path


if __name__ == "__main__":
    path = download_and_trim()
    print(f"Wrote trimmed clip to {path}")
