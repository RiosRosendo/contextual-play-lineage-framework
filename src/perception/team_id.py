"""Team identification by jersey color clustering, per the project spec section 4
(Layer 1). Runs k-means (k=2) over the mean torso-crop color of all player
detections in a frame/batch, and labels each detection by nearest cluster.
The color_detector fallback already provides a team hint directly (used
as-is); this module is what the YOLO path needs since COCO gives no team
information.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def torso_crop_mean_color(frame_bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    tx1, tx2 = int(max(0, x1)), int(min(w, x2))
    ty1 = int(max(0, y1 + 0.25 * (y2 - y1)))
    ty2 = int(min(h, y1 + 0.6 * (y2 - y1)))
    if tx2 <= tx1 or ty2 <= ty1:
        return np.array([0.0, 0.0, 0.0])
    crop = frame_bgr[ty1:ty2, tx1:tx2]
    return crop.reshape(-1, 3).mean(axis=0)


def assign_teams(torso_colors: list[np.ndarray]) -> list[int]:
    """Clusters torso colors into 2 teams. Returns a cluster label (0/1) per
    input color. With < 2 samples, everyone gets label 0.

    Stateless -- cluster label 0 vs 1 is arbitrary each call, with no
    memory of which real jersey color "0" meant last time. Fine for a
    single frame/batch in isolation, but calling this repeatedly (once per
    frame, or once per shot after a camera cut) gives no guarantee that
    label 0 keeps meaning the same real team across calls. Real footage
    validation (Sunderland vs Liverpool, see the dev log) found this
    matters: a 100s clip with 9 camera cuts re-clustered blind on every
    frame, so "team_a" wasn't guaranteed to mean the same real team after
    a cut. Use TeamColorAnchor instead when identity must persist across
    frames/shots."""
    if len(torso_colors) < 2:
        return [0] * len(torso_colors)
    x = np.stack(torso_colors)
    labels = KMeans(n_clusters=2, n_init=4, random_state=0).fit_predict(x)
    return labels.tolist()


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class TeamColorAnchor:
    """Persists team-color identity across frames and shots: "team_a"
    keeps meaning the same real jersey color throughout a whole clip,
    instead of being re-clustered blind every call the way `assign_teams`
    is. Bootstraps its 2 reference colors via k-means the first time it
    sees enough players, then classifies every subsequent call by nearest
    reference centroid (not by re-clustering), with slow exponential
    adaptation so gradual lighting drift doesn't require a fresh anchor.
    Deliberately does NOT reset on a scene cut -- that's the point: the
    caller creates one instance for a whole clip and keeps using it across
    shots, which is what actually fixes the bug described in
    `assign_teams`'s docstring.

    Real-footage validation (the dev log, 2026-07-15) found a second bug
    this class needs to guard against: when two players' boxes overlap
    (occlusion, a tangle, a tackle), the torso crop can pick up a blend of
    both players' colors, producing an ambiguous sample that still gets
    confidently assigned to *some* team and then folds into that team's
    running centroid via the EMA update -- observed directly to flip a
    single track's reported team mid-life (e.g. "team_a" on one frame,
    "team_b" on the very next, same real player). A sample is now treated
    as untrustworthy -- classified as `None` (no team) rather than forced
    to a guess, and excluded from the centroid update -- if either: (1) it
    isn't clearly closer to one reference centroid than the other (a
    minimum-separation ratio), or (2) its box significantly overlaps
    another detected player's box this frame (occlusion). `None` is
    already a valid, handled value for `team` everywhere downstream (ball
    and referee rows already carry it), so an honest "don't know" for one
    frame is a normal case, not a new failure mode for callers to handle."""

    def __init__(self, ema_alpha: float = 0.05, min_separation_ratio: float = 1.2,
                 overlap_iou_threshold: float = 0.1):
        self.centroids: np.ndarray | None = None  # shape (2, 3), BGR
        self.ema_alpha = ema_alpha
        # Nearest centroid must be at least this much closer than the other
        # for a sample to count as confidently classified -- 1.2 means the
        # farther centroid must be >=20% more distant than the nearest one.
        self.min_separation_ratio = min_separation_ratio
        self.overlap_iou_threshold = overlap_iou_threshold

    def _occluded_mask(self, boxes: list[tuple] | None, n: int) -> np.ndarray:
        if not boxes:
            return np.zeros(n, dtype=bool)
        occluded = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if _iou(boxes[i], boxes[j]) > self.overlap_iou_threshold:
                    occluded[i] = occluded[j] = True
        return occluded

    def assign(self, torso_colors: list[np.ndarray], boxes: list[tuple] | None = None) -> list[int | None]:
        """`boxes` (same order/length as `torso_colors`, each an (x1,y1,x2,y2)
        tuple) is optional but should be passed whenever available -- it's
        what makes the occlusion check possible. Without it, only the
        minimum-separation check applies."""
        if not torso_colors:
            return []
        x = np.stack(torso_colors)
        n = len(torso_colors)
        occluded = self._occluded_mask(boxes, n)

        if self.centroids is None:
            if n < 2:
                return [0] * n
            trusted = x[~occluded] if np.any(~occluded) else x
            labels_trusted = KMeans(n_clusters=2, n_init=4, random_state=0).fit_predict(trusted)
            self.centroids = np.array([
                trusted[labels_trusted == k].mean(axis=0) if np.any(labels_trusted == k) else trusted.mean(axis=0)
                for k in (0, 1)
            ])

        # Re-derived from the (possibly just-bootstrapped) centroids either
        # way, so occluded/bootstrap samples get a distance-based label/None
        # consistently with every later call, not a first-frame special case.
        dists = np.linalg.norm(x[:, None, :] - self.centroids[None, :, :], axis=2)
        d_sorted = np.sort(dists, axis=1)
        confident = d_sorted[:, 1] >= self.min_separation_ratio * np.maximum(d_sorted[:, 0], 1e-6)
        trust_sample = confident & ~occluded
        nearest = dists.argmin(axis=1)

        for k in (0, 1):
            assigned = x[trust_sample & (nearest == k)]
            if len(assigned):
                self.centroids[k] = (1 - self.ema_alpha) * self.centroids[k] + self.ema_alpha * assigned.mean(axis=0)

        return [int(nearest[i]) if trust_sample[i] else None for i in range(n)]
