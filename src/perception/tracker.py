"""Minimal centroid tracker to assign persistent IDs across frames. Stands in
for ByteTrack/BoT-SORT (CLAUDE.md section 4) for the first end-to-end pass --
greedy nearest-neighbor matching per class, no re-identification after a
track is lost for more than `max_missed` frames. Swap-in of a real tracker is
tracked in TODO.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Track:
    track_id: int
    cls: str
    team: str | None
    x: float
    y: float
    missed: int = 0


class CentroidTracker:
    def __init__(self, max_match_dist_m: float = 3.0, max_missed: int = 5):
        self.max_match_dist_m = max_match_dist_m
        self.max_missed = max_missed
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: list[dict]) -> list[dict]:
        """detections: list of {cls, team, x, y, conf}. Returns the same
        dicts with a 'track_id' key added, matched against existing tracks
        of the same class."""
        unmatched_track_ids = set(self._tracks.keys())
        results = []

        for det in detections:
            best_id, best_dist = None, self.max_match_dist_m
            for tid in unmatched_track_ids:
                track = self._tracks[tid]
                if track.cls != det["cls"]:
                    continue
                dist = ((track.x - det["x"]) ** 2 + (track.y - det["y"]) ** 2) ** 0.5
                if dist < best_dist:
                    best_id, best_dist = tid, dist

            if best_id is not None:
                track = self._tracks[best_id]
                track.x, track.y, track.missed = det["x"], det["y"], 0
                if det.get("team"):
                    track.team = det["team"]
                unmatched_track_ids.discard(best_id)
                tid = best_id
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = Track(tid, det["cls"], det.get("team"), det["x"], det["y"])

            results.append({**det, "track_id": tid, "team": self._tracks[tid].team})

        for tid in unmatched_track_ids:
            self._tracks[tid].missed += 1
        self._tracks = {
            tid: t for tid, t in self._tracks.items() if t.missed <= self.max_missed
        }
        return results
