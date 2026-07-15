"""Validates src/perception/scene_cut.py against the real SoccerNet clip,
where a genuine camera cut was found (a wide tactical shot cut to a tight
player-reaction close-up around frame 69-131) alongside a case that looks
similar in raw detection-count terms but isn't a cut at all (frames
~386-404: heavy motion blur from a fast camera pan, still the same wide
shot). Reports whether the detector correctly flags the real cut and its
false-positive rate on the non-cut portions, per Rosendo's request.

Usage:
    python -m notebooks.validate_scene_cut
"""
from __future__ import annotations

import cv2

from src.perception.scene_cut import CORREL_CUT_THRESHOLD, _frame_histogram, detect_cuts

CLIP_PATH = "data/raw/soccernet/clip20s.mp4"
KNOWN_CUT_REGION = (69, 132)  # ground truth: established by frame-by-frame detection counts,
                              # confirmed visually as a genuine cut to a close-up
KNOWN_PAN_REGION = (386, 405)  # ground truth: fast pan/motion blur, same shot, NOT a cut


def main() -> None:
    cuts = detect_cuts(CLIP_PATH)
    print(f"Detected {len(cuts)} cut(s):")
    for c in cuts:
        print(f"  frame {c.frame_idx}, correlation={c.correlation:.3f}")

    cut_frames = {c.frame_idx for c in cuts}
    hit_start = KNOWN_CUT_REGION[0] in cut_frames
    hit_end = KNOWN_CUT_REGION[1] in cut_frames
    print(f"\nCorrectly flagged start of known cut ({KNOWN_CUT_REGION[0]}): {hit_start}")
    print(f"Correctly flagged end of known cut / cut back ({KNOWN_CUT_REGION[1]}): {hit_end}")

    false_positives_in_pan_region = [
        c for c in cuts if KNOWN_PAN_REGION[0] <= c.frame_idx < KNOWN_PAN_REGION[1]
    ]
    print(f"\nFalse positives in the known non-cut (motion-blur/pan) region "
          f"{KNOWN_PAN_REGION}: {len(false_positives_in_pan_region)}")

    other_false_positives = [
        c for c in cuts
        if not (KNOWN_CUT_REGION[0] <= c.frame_idx <= KNOWN_CUT_REGION[1])
    ]
    print(f"Cuts flagged outside the known real cut entirely (false positives "
          f"anywhere else in the clip): {len(other_false_positives)}")

    # margin: how close was the nearest non-flagged frame to the threshold?
    cap = cv2.VideoCapture(CLIP_PATH)
    prev_hist, closest_margin, i = None, None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hist = _frame_histogram(frame)
        if prev_hist is not None:
            correl = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if correl >= CORREL_CUT_THRESHOLD:
                margin = correl - CORREL_CUT_THRESHOLD
                if closest_margin is None or margin < closest_margin:
                    closest_margin = margin
        prev_hist = hist
        i += 1
    cap.release()
    print(f"\nClosest a non-cut frame came to the threshold ({CORREL_CUT_THRESHOLD}): "
          f"{closest_margin:.3f} above it.")


if __name__ == "__main__":
    main()
