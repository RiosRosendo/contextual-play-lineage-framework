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
    input color. With < 2 samples, everyone gets label 0."""
    if len(torso_colors) < 2:
        return [0] * len(torso_colors)
    x = np.stack(torso_colors)
    labels = KMeans(n_clusters=2, n_init=4, random_state=0).fit_predict(x)
    return labels.tolist()
