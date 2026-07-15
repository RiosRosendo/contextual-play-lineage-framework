"""Team identification by jersey color clustering, per CLAUDE.md section 4
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
    validation (Sunderland vs Liverpool, see PROGRESS.md) found this
    matters: a 100s clip with 9 camera cuts re-clustered blind on every
    frame, so "team_a" wasn't guaranteed to mean the same real team after
    a cut. Use TeamColorAnchor instead when identity must persist across
    frames/shots."""
    if len(torso_colors) < 2:
        return [0] * len(torso_colors)
    x = np.stack(torso_colors)
    labels = KMeans(n_clusters=2, n_init=4, random_state=0).fit_predict(x)
    return labels.tolist()


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
    `assign_teams`'s docstring."""

    def __init__(self, ema_alpha: float = 0.05):
        self.centroids: np.ndarray | None = None  # shape (2, 3), BGR
        self.ema_alpha = ema_alpha

    def assign(self, torso_colors: list[np.ndarray]) -> list[int]:
        if not torso_colors:
            return []
        x = np.stack(torso_colors)

        if self.centroids is None:
            if len(torso_colors) < 2:
                return [0] * len(torso_colors)
            labels = KMeans(n_clusters=2, n_init=4, random_state=0).fit_predict(x)
            self.centroids = np.array([
                x[labels == k].mean(axis=0) if np.any(labels == k) else x.mean(axis=0)
                for k in (0, 1)
            ])
            return labels.tolist()

        dists = np.linalg.norm(x[:, None, :] - self.centroids[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        for k in (0, 1):
            assigned = x[labels == k]
            if len(assigned):
                self.centroids[k] = (1 - self.ema_alpha) * self.centroids[k] + self.ema_alpha * assigned.mean(axis=0)
        return labels.tolist()
