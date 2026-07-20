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
    is. Bootstraps 3 reference colors via k-means (2 teams + referee, see
    below) the first time it sees enough players, then classifies every
    subsequent call by nearest reference centroid (not by re-clustering),
    with slow exponential adaptation so gradual lighting drift doesn't
    require a fresh anchor. Deliberately does NOT reset on a scene cut --
    that's the point: the caller creates one instance for a whole clip and
    keeps using it across shots, which is what actually fixes the bug
    described in `assign_teams`'s docstring.

    Referee discrimination (2026-07-17): bootstraps with k=3 clusters
    instead of 2 -- a referee's kit is conventionally a solid color
    distinct from both teams' (the classic reason referees wear black or a
    bright singleton color), so it should separate out as its own cluster
    the same way the two team kits do. Since color alone doesn't say
    *which* of the 3 clusters is the referee (a dark team kit looks the
    same as a dark referee kit to k-means), cluster POPULATION size is
    used to decide: the referee is one person against ~10-11 per team, so
    whichever of the 3 bootstrap clusters has the fewest members is
    labeled referee, and the other two become team_a/team_b as before.
    This is a real, disclosed assumption -- it fails if the bootstrap
    frame doesn't have enough of a mixed sample (e.g. a tight shot with
    only 2-3 people, or a frame where the referee happens to be one of
    several similarly-dark-kitted outfield players) -- but it needs no new
    model and reuses the exact color-clustering infrastructure already
    validated for team identity.

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

    MIN_BOOTSTRAP_SAMPLES = 4  # need enough people to safely split into 2 teams + a referee singleton

    def __init__(self, ema_alpha: float = 0.05, min_separation_ratio: float = 1.2,
                 overlap_iou_threshold: float = 0.1):
        self.centroids: np.ndarray | None = None  # shape (3, 3), BGR: [team_a, team_b, referee]
        self.ema_alpha = ema_alpha
        # Nearest centroid must be at least this much closer than the next-
        # nearest for a sample to count as confidently classified -- 1.2
        # means the next-nearest must be >=20% more distant than the
        # nearest one.
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

    def assign(self, torso_colors: list[np.ndarray], boxes: list[tuple] | None = None,
               patterned: list[bool] | None = None) -> list[int | None]:
        """`boxes` (same order/length as `torso_colors`, each an (x1,y1,x2,y2)
        tuple) is optional but should be passed whenever available -- it's
        what makes the occlusion check possible. Without it, only the
        minimum-separation check applies.

        `patterned` (2026-07-19, same order/length as `torso_colors`) is a
        hard domain constraint: real referees never wear striped/patterned
        kits (IFAB Law 4), so a crop flagged `True` here can NEVER be
        classified as referee (label 2), regardless of its color-distance
        to the referee centroid -- confirmed necessary directly, not
        assumed: Athletic Bilbao's and Hull City's striped kits were both
        found confidently misclassified as referee purely on color
        distance, on two independent real clips. See
        `kit_pattern_classifier.py` for why this is a trained classifier
        rather than a hand-crafted statistic (5 cheap proxies tried and
        empirically failed to separate genuine referees from patterned
        kits first). Applied at BOTH bootstrap (so a patterned team's
        fragment can't be mistaken for the referee cluster in the first
        place) and at per-frame classification (so a patterned sample is
        never assigned to the referee cluster afterward either) --
        omitting either half would leave the other route open.

        Returns, per input color: 0 (team_a), 1 (team_b), 2 (referee), or
        None -- either not enough trusted samples exist yet to bootstrap
        (fewer than MIN_BOOTSTRAP_SAMPLES), or this specific sample isn't
        trustworthy (ambiguous or occluded, see the class docstring)."""
        if not torso_colors:
            return []
        x = np.stack(torso_colors)
        n = len(torso_colors)
        occluded = self._occluded_mask(boxes, n)
        patterned_arr = np.array(patterned, dtype=bool) if patterned is not None else np.zeros(n, dtype=bool)

        if self.centroids is None:
            trust_mask = ~occluded
            trusted = x[trust_mask] if np.any(trust_mask) else x
            trusted_patterned = patterned_arr[trust_mask] if np.any(trust_mask) else patterned_arr
            if len(trusted) < self.MIN_BOOTSTRAP_SAMPLES:
                return [None] * n  # not enough people yet to safely separate 2 teams + a referee
            labels_trusted = KMeans(n_clusters=3, n_init=4, random_state=0).fit_predict(trusted)
            counts = [int(np.sum(labels_trusted == k)) for k in range(3)]
            # The referee is 1 person against ~10-11 per team in a typical
            # bootstrap sample -- color alone can't say *which* cluster is
            # the referee (a dark team kit clusters the same as a dark
            # referee kit), but population size can (see class docstring).
            # Population-size candidates are tried smallest-first, but any
            # cluster that's MOSTLY patterned samples is skipped -- it's
            # more likely a small-in-this-sample fragment of a patterned
            # team than genuine referee crops (a patterned team's own
            # smallest subset, not a real singleton referee).
            by_size = sorted(range(3), key=lambda k: counts[k])
            referee_cluster = None
            for k in by_size:
                cluster_patterned_frac = trusted_patterned[labels_trusted == k].mean() if counts[k] else 1.0
                if cluster_patterned_frac < 0.5:
                    referee_cluster = k
                    break
            if referee_cluster is None:
                referee_cluster = by_size[0]  # rare: every cluster looks patterned -- fall back rather than never bootstrap
            team_clusters = [k for k in range(3) if k != referee_cluster]
            self.centroids = np.array([
                trusted[labels_trusted == team_clusters[0]].mean(axis=0),
                trusted[labels_trusted == team_clusters[1]].mean(axis=0),
                trusted[labels_trusted == referee_cluster].mean(axis=0),
            ])

        # Re-derived from the (possibly just-bootstrapped) centroids either
        # way, so occluded/bootstrap samples get a distance-based label/None
        # consistently with every later call, not a first-frame special case.
        dists = np.linalg.norm(x[:, None, :] - self.centroids[None, :, :], axis=2)
        d_sorted = np.sort(dists, axis=1)
        confident = d_sorted[:, 1] >= self.min_separation_ratio * np.maximum(d_sorted[:, 0], 1e-6)
        trust_sample = confident & ~occluded
        nearest = dists.argmin(axis=1)

        # Hard veto: a patterned sample nearest to the referee centroid is
        # reassigned to whichever TEAM centroid (0/1) is actually closer --
        # it must never be allowed to read as referee, per this function's
        # own docstring.
        vetoed = patterned_arr & (nearest == 2)
        if np.any(vetoed):
            nearest[vetoed] = dists[vetoed][:, :2].argmin(axis=1)

        for k in range(3):
            assigned = x[trust_sample & (nearest == k)]
            if len(assigned):
                self.centroids[k] = (1 - self.ema_alpha) * self.centroids[k] + self.ema_alpha * assigned.mean(axis=0)

        return [int(nearest[i]) if trust_sample[i] else None for i in range(n)]
