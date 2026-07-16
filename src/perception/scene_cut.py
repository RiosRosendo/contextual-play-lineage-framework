"""Cheap scene-cut (camera angle change) detector. High priority per
Rosendo: real broadcast footage cuts between camera angles within seconds
(confirmed directly on the SoccerNet clip -- see the dev log), and both
calibration (one homography assumed per clip) and tracking (identities
assumed continuous) silently break across a cut with no warning. This
flags cut points so callers can split a clip into per-shot segments before
trusting either.

Uses HSV histogram correlation between consecutive frames, not raw pixel
difference. This distinguishes a real cut from a fast camera pan/whip
within the same continuous shot -- a case that came up directly in the
same clip (frames ~386-404: heavy motion blur from a fast pan, detection
count dropped similarly to the real cut, but it's still the same wide
pitch shot). A pan/whip still keeps the same dominant colors (pitch green,
crowd, sky) just spatially shifted and blurred, so its histogram stays
well-correlated frame to frame; a real cut to a different shot (e.g. a
player close-up, dominated by skin/jersey colors instead of pitch green)
does not. Raw frame-difference (mean absolute pixel difference) does not
make this distinction -- both a whip pan and a real cut produce a large
frame-to-frame pixel difference -- which is why histogram correlation is
used instead, not in addition, as the primary signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CORREL_CUT_THRESHOLD = 0.7  # cv2.HISTCMP_CORREL; 1.0 = identical, lower = more different
HIST_SIZE = (30, 32)  # (hue bins, saturation bins)
RESIZE_TO = (64, 36)  # downsample before histogramming -- this only needs to be cheap and global


def _frame_histogram(frame_bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame_bgr, RESIZE_TO, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HIST_SIZE), [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


@dataclass
class CutPoint:
    frame_idx: int
    correlation: float  # with the previous frame; lower means more different


def detect_cuts(video_path: str, threshold: float = CORREL_CUT_THRESHOLD) -> list[CutPoint]:
    """Returns the frame index of each detected cut (the first frame of the
    new shot). A cut at frame_idx means frames [0, frame_idx) and
    [frame_idx, ...) should be treated as separate shots -- recompute
    calibration and reset tracking at that boundary."""
    cap = cv2.VideoCapture(video_path)
    cuts: list[CutPoint] = []
    prev_hist = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hist = _frame_histogram(frame)
        if prev_hist is not None:
            correl = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if correl < threshold:
                cuts.append(CutPoint(frame_idx, correl))
        prev_hist = hist
        frame_idx += 1
    cap.release()
    return cuts


def split_into_shots(video_path: str, threshold: float = CORREL_CUT_THRESHOLD) -> list[tuple[int, int]]:
    """Returns a list of (start_frame, end_frame_exclusive) shot segments
    covering the whole video, split at detected cuts."""
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cut_frames = [c.frame_idx for c in detect_cuts(video_path, threshold)]
    bounds = [0] + cut_frames + [n_frames]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
