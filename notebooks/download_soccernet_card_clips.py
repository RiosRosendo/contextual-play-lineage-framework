"""Downloads and trims a short (20s) real clip around each of several
isolated, single-card incidents found by notebooks/find_single_card_incidents.py,
across different real matches -- building a small, diverse test set for the
foul-verdict-audit capability (src/assistant/explain.py) beyond the single
Sunderland-Liverpool anecdote already validated.

Same download discipline as notebooks/download_soccernet_clip.py: only the
specific half needed is downloaded, and the full (~1GB) half is deleted
immediately after trimming -- only the short clip is kept locally
(gitignored, not pushed, same as every other real SoccerNet clip in this
project).

Usage:
    python -m notebooks.download_soccernet_card_clips
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

# Selected from find_single_card_incidents.py's output: isolated single-card
# incidents (no other card within 20s), one per match, spread across
# different clubs/matches -- avoiding the already-used Sunderland-Liverpool
# double-yellow-card moment.
CARD_JOBS = [
    {
        "slug": "chelsea_burnley",
        "game": "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley",
        "half": 1, "card_time_s": 40 * 60 + 8, "label": "y-card", "team": "away",
    },
    {
        "slug": "palace_arsenal",
        "game": "england_epl/2014-2015/2015-02-21 - 18-00 Crystal Palace 1 - 2 Arsenal",
        "half": 1, "card_time_s": 18 * 60 + 33, "label": "y-card", "team": "home",
    },
    {
        "slug": "swansea_manutd",
        "game": "england_epl/2014-2015/2015-02-21 - 18-00 Swansea 2 - 1 Manchester United",
        "half": 2, "card_time_s": 2 * 60 + 43, "label": "y-card", "team": "away",
    },
    {
        "slug": "southampton_liverpool",
        "game": "england_epl/2014-2015/2015-02-22 - 19-15 Southampton 0 - 2 Liverpool",
        "half": 2, "card_time_s": 17 * 60 + 55, "label": "y-card", "team": "home",
    },
    {
        "slug": "mancity_watford",
        "game": "england_epl/2015-2016/2015-08-29 - 17-00 Manchester City 2 - 0 Watford",
        "half": 1, "card_time_s": 39 * 60 + 22, "label": "y-card", "team": "home",
    },
]


def download_and_trim(job: dict) -> str:
    load_dotenv()
    password = os.environ.get("SOCCERNET_PASSWORD")
    if not password:
        raise RuntimeError("SOCCERNET_PASSWORD not set. Add it to a local .env file.")

    from SoccerNet.Downloader import SoccerNetDownloader

    half_file = f"{job['half']}_720p.mkv"
    out_path = f"{DEST_DIR}/card_{job['slug']}.mp4"

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

    shutil.rmtree(os.path.join(DEST_DIR, job["game"].split("/")[0]))  # drop the ~1GB source
    return out_path


if __name__ == "__main__":
    for job in CARD_JOBS:
        print(f"Downloading/trimming {job['slug']} ({job['game']}, half {job['half']}, "
              f"card at {job['card_time_s']}s)...")
        path = download_and_trim(job)
        print(f"  -> wrote {path}")
