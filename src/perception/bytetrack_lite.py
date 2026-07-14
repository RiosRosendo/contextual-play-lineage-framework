"""Two-stage IoU tracker inspired by ByteTrack (Zhang et al., 2022), used in
place of Ultralytics' built-in ByteTrack/BoT-SORT for the YOLO backend.

CLAUDE.md section 4 names ByteTrack/BoT-SORT specifically, and both were
tried first via `model.track(..., tracker="bytetrack.yaml" / "botsort.yaml")`.
Both showed the same bug in this environment: per-frame detection counts
collapsed from ~15-20 (correct, matches plain `model.predict()` on the same
frames) to 1-4 boxes, with periodic frames where every box lost its track
id entirely. This reproduced identically with both trackers and with both
our fine-tuned model and the plain pretrained COCO model, so it's an
Ultralytics/lap library issue in this environment, not something specific
to our model or code. A numpy downgrade was floated as a possible ABI-
mismatch fix but reverted immediately since it breaks opencv's own numpy
>=2 requirement. See PROGRESS.md "Tracker replacement" entry for the full
investigation.

This module keeps ByteTrack's core idea (which is the actual point of using
it over naive nearest-neighbor matching): detections are matched to
existing tracks in two confidence tiers -- high-confidence detections are
matched first via Hungarian/IoU assignment and may start new tracks when
unmatched; low-confidence detections are only used to extend already-
matched tracks through occlusion/motion blur and never spawn a new track.
Lost tracks are kept alive (unmatched, not emitted) for a short buffer
window in case they reappear. What's simplified relative to real ByteTrack:
instead of a full Kalman filter, motion is predicted with a constant-
velocity estimate (last frame-to-frame box shift) and matching is done
against that predicted box, not the track's last raw position. This
matters more than it sounds: an early version without any motion
prediction fragmented the ball (small, ~8x8px, moving ~6-8px/frame) into a
new track almost every single frame -- consecutive raw boxes had IoU
~0.10, well under the 0.3 match threshold, even though the ball was
tracked correctly moment to moment. Predicting where the box should be
based on its last velocity brings that IoU back above threshold. Slower
objects (players) were largely unaffected either way.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

HIGH_CONF_THRESH = 0.5
IOU_MATCH_THRESH = 0.3
TRACK_BUFFER_FRAMES = 30
# The ball is a much smaller box than a player (often <10x10px) and can
# accelerate sharply (a kick), so even good velocity prediction leaves more
# residual pixel error relative to its own box size than for a player. A
# stricter shared IoU threshold made it lose its track on almost every frame
# of a fast shot sequence. A looser threshold for small/fast classes is a
# common practical adjustment in IoU trackers; there isn't a similarly small,
# fast class among people, so this doesn't loosen matching for players/referee.
IOU_MATCH_THRESH_BY_CLASS = {"ball": 0.1}


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _predicted_box(box: tuple, velocity: tuple) -> tuple:
    dx, dy = velocity
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


@dataclass
class _Track:
    track_id: int
    cls: str
    box: tuple
    team: str | None = None
    missed: int = 0
    velocity: tuple = (0.0, 0.0)  # constant-velocity motion estimate, in px/frame

    def predicted_box(self) -> tuple:
        return _predicted_box(self.box, self.velocity)


@dataclass
class ByteTrackLite:
    high_conf_thresh: float = HIGH_CONF_THRESH
    iou_match_thresh: float = IOU_MATCH_THRESH
    track_buffer: int = TRACK_BUFFER_FRAMES
    _tracks: dict = field(default_factory=dict)
    _next_id: int = 1

    def _match_tier(self, dets: list[dict], candidate_track_ids: list[int],
                     allow_new: bool) -> tuple[dict, list[int]]:
        """Matches a confidence tier of detections against the given
        candidate tracks (same class only). Returns {det_index: track_id}
        and the list of still-unmatched candidate track ids."""
        assigned: dict[int, int] = {}
        remaining = set(candidate_track_ids)

        by_cls: dict[str, list[int]] = {}
        for i, d in enumerate(dets):
            by_cls.setdefault(d["cls"], []).append(i)

        for cls, det_idxs in by_cls.items():
            cls_track_ids = [tid for tid in remaining if self._tracks[tid].cls == cls]
            if det_idxs and cls_track_ids:
                cost = np.ones((len(det_idxs), len(cls_track_ids)))
                for r, di in enumerate(det_idxs):
                    for c, tid in enumerate(cls_track_ids):
                        cost[r, c] = 1.0 - _iou(dets[di]["box"], self._tracks[tid].predicted_box())
                rows, cols = linear_sum_assignment(cost)
                thresh = IOU_MATCH_THRESH_BY_CLASS.get(cls, self.iou_match_thresh)
                for r, c in zip(rows, cols):
                    if cost[r, c] <= 1.0 - thresh:
                        di, tid = det_idxs[r], cls_track_ids[c]
                        assigned[di] = tid
                        remaining.discard(tid)

            if allow_new:
                for di in det_idxs:
                    if di not in assigned:
                        tid = self._next_id
                        self._next_id += 1
                        self._tracks[tid] = _Track(tid, dets[di]["cls"], dets[di]["box"])
                        assigned[di] = tid

        return assigned, list(remaining)

    def _apply_match(self, tid: int, new_box: tuple) -> None:
        track = self._tracks[tid]
        track.velocity = (new_box[0] - track.box[0], new_box[1] - track.box[1])
        track.box = new_box
        track.missed = 0

    def update(self, detections: list[dict]) -> list[dict]:
        """detections: list of {cls, box: (x1,y1,x2,y2), conf, team (optional)}.
        Returns the same dicts with a 'track_id' key added -- one row per
        input detection (order preserved)."""
        high = [d for d in detections if d["conf"] >= self.high_conf_thresh]
        low = [d for d in detections if d["conf"] < self.high_conf_thresh]

        all_track_ids = list(self._tracks.keys())
        high_assigned, unmatched_after_high = self._match_tier(high, all_track_ids, allow_new=True)
        low_assigned, unmatched_after_low = self._match_tier(low, unmatched_after_high, allow_new=False)

        results = [None] * len(detections)
        high_indices = [i for i, d in enumerate(detections) if d["conf"] >= self.high_conf_thresh]
        low_indices = [i for i, d in enumerate(detections) if d["conf"] < self.high_conf_thresh]

        for local_i, global_i in enumerate(high_indices):
            tid = high_assigned[local_i]
            self._apply_match(tid, detections[global_i]["box"])
            results[global_i] = {**detections[global_i], "track_id": tid}
        for local_i, global_i in enumerate(low_indices):
            if local_i in low_assigned:
                tid = low_assigned[local_i]
                self._apply_match(tid, detections[global_i]["box"])
                results[global_i] = {**detections[global_i], "track_id": tid}

        matched_ids = {r["track_id"] for r in results if r is not None}
        for tid in unmatched_after_low:
            self._tracks[tid].missed += 1
        self._tracks = {
            tid: t for tid, t in self._tracks.items()
            if tid in matched_ids or t.missed <= self.track_buffer
        }

        return [r for r in results if r is not None]
