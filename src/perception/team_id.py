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
    frame is a normal case, not a new failure mode for callers to handle.

    Referee-bootstrap contamination (2026-07-20): real-footage validation
    (Chelsea-Burnley) found the population-size referee heuristic above
    fails hard when the actual referee simply isn't in the bootstrap
    frame at all -- confirmed directly by instrumenting this class: frame
    0 held 8 people, all outfield players (4 v 4), yet k=3 still forced a
    3-way split, and the smallest cluster (population 1) turned out to be
    a real Burnley player who happened to read as a color outlier, not a
    referee. His kit color then permanently seeded the referee centroid,
    misclassifying genuine teammates throughout the rest of the clip
    (confirmed on 3 independently-sampled tracks) while the real referee
    (all-black kit, never captured as any reference color) was
    misclassified onto a team instead. A second, independent instance of
    the same mechanism was found on `foul_arsenal_anderlecht.mp4` (there
    it happened to self-correct instead of persisting, previously
    mis-attributed to generic "bootstrap instability").

    Population size alone can't tell a genuine referee singleton apart
    from a real team's own color-sampling outlier, but color
    DISTINCTIVENESS can: a referee's kit is chosen specifically to not
    clash with either team's, so a genuine referee cluster sits far from
    BOTH team clusters -- typically farther than the two team clusters
    sit from each other. A same-team outlier does not: checked directly
    across all 6 locally-downloaded clips' own bootstrap frames, the
    smallest cluster's distance to the nearer of the other two, as a
    fraction of those other two's own mutual distance, is 1.683/0.838 on
    Leicester/Swansea (both confirmed-correct real referee identifications)
    versus 0.667/0.514 on Chelsea-Burnley/Arsenal-Anderlecht (both
    confirmed-wrong, a same-team outlier) -- a clean gap between the two
    groups. `REFEREE_MIN_DISTINCTIVENESS_RATIO` sits between them. A
    candidate cluster/sample is only ever accepted as the referee if it
    clears both this ratio AND the existing patterned-kit veto; when
    nothing clears it yet, referee identification is deferred (team_a/
    team_b bootstrap proceeds via k=2 alone, so team classification isn't
    held up) and retried on every subsequent frame's samples against the
    now-fixed team centroids, until some frame naturally contains a
    confidently-distinct referee color (or never does, for the whole
    clip -- an honest `None` for every frame rather than a permanent wrong
    guess, consistent with this class's existing abstention philosophy)."""

    MIN_BOOTSTRAP_SAMPLES = 4  # need enough people to safely split into 2 teams + a referee singleton
    REFEREE_MIN_DISTINCTIVENESS_RATIO = 0.8  # see class docstring for the 6-clip calibration behind this cut
    # A second, real bug found on card_hull_arsenal.mp4 (2026-07-20): the
    # ratio above is only meaningful when the two things being compared
    # (the "other two" clusters, or the two team centroids) are themselves
    # a reliable baseline. On a small/noisy bootstrap sample (this clip's
    # first frame had only 4 trusted people, right at MIN_BOOTSTRAP_SAMPLES,
    # with most of Hull's own patterned kit excluded from team formation
    # entirely), the two resulting team centroids ended up only 31.0 apart
    # -- close enough that an ORDINARY Arsenal player's own color still
    # cleared the 0.8 ratio (25.4/31.0=0.819) purely because the
    # denominator was small, not because he was genuinely referee-colored.
    # Checked directly against every confirmed-good case: Leicester 38.7,
    # clip20s 66.1, Chelsea-Burnley 70.8, Swansea 94.4 -- all comfortably
    # separated from Hull-Arsenal's 31.0. This absolute floor is a second,
    # independent guard alongside the ratio -- both must hold. Applied to
    # the one-shot k=3 bootstrap attempt, where a full 3-cluster fit is
    # already a fairly reliable computation in its own right.
    MIN_TEAM_SEPARATION_FOR_REFEREE_CHECK = 35.0
    # The ongoing per-frame deferred retry (below) needs a stricter floor
    # than the one-shot bootstrap test: checked directly on Hull-Arsenal,
    # team_centroids crossed 35.0 (the bootstrap floor) within ~1.4s of
    # real time, but were still noisy enough (d_teams=36.1) that an
    # ordinary Arsenal player passed the ratio there too -- a second,
    # independent misclassification via the deferred path specifically,
    # not the one-shot bootstrap path. Raising this deferred-only floor to
    # 60.0 (team centroids given more real time/samples to mature via EMA
    # before ever being trusted as a baseline) eliminated both false
    # positives, at the honest cost of this specific clip's own referee
    # almost never being confidently confirmed at all (a close-up-framing
    # issue independent of this floor, see the module's dev notes) --
    # preferred over confidently mislabeling real players, per this
    # class's abstention philosophy.
    MIN_TEAM_SEPARATION_FOR_DEFERRED_CHECK = 60.0

    def __init__(self, ema_alpha: float = 0.05, min_separation_ratio: float = 1.2,
                 overlap_iou_threshold: float = 0.1):
        # team_centroids: shape (2, 3) BGR [team_a, team_b], set once both
        # teams bootstrap. referee_centroid: shape (3,) BGR, or None until
        # a confidently-distinct referee color is found (see docstring) --
        # no referee label (2) is ever returned by `assign` until then.
        self.team_centroids: np.ndarray | None = None
        self.referee_centroid: np.ndarray | None = None
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
        team_a/team_b (fewer than MIN_BOOTSTRAP_SAMPLES), this specific
        sample isn't trustworthy (ambiguous or occluded, see the class
        docstring), or a referee color hasn't been confidently identified
        yet (see the class docstring's 2026-07-20 note) -- label 2 is
        never returned until then."""
        if not torso_colors:
            return []
        x = np.stack(torso_colors)
        n = len(torso_colors)
        occluded = self._occluded_mask(boxes, n)
        patterned_arr = np.array(patterned, dtype=bool) if patterned is not None else np.zeros(n, dtype=bool)

        if self.team_centroids is None:
            trust_mask = ~occluded
            trusted = x[trust_mask] if np.any(trust_mask) else x
            trusted_patterned = patterned_arr[trust_mask] if np.any(trust_mask) else patterned_arr
            if len(trusted) < self.MIN_BOOTSTRAP_SAMPLES:
                return [None] * n  # not enough people yet to safely bootstrap even 2 teams
            labels_trusted = KMeans(n_clusters=3, n_init=4, random_state=0).fit_predict(trusted)
            counts = [int(np.sum(labels_trusted == k)) for k in range(3)]
            centroids_trusted = [trusted[labels_trusted == k].mean(axis=0) for k in range(3)]
            # The referee is 1 person against ~10-11 per team in a typical
            # bootstrap sample -- population size alone can't say *which*
            # cluster is the referee though (a real team's own color
            # outlier can just as easily be the smallest cluster -- see the
            # class docstring's Chelsea-Burnley/Arsenal-Anderlecht
            # evidence). Only the single smallest-population cluster is
            # ever considered as a referee candidate (falling through to
            # the next-smallest ONLY if the smallest looks patterned --
            # a patterned team's own small fragment, not a real referee --
            # never for any other reason). A first real bug found and
            # fixed here (2026-07-20): testing progressively LARGER
            # clusters as fallback referee candidates once the true
            # smallest cluster fails on other grounds is unsound -- on
            # Chelsea-Burnley, once the real 1-person outlier correctly
            # failed the distinctiveness check below, an earlier version
            # of this loop went on to test the 3-person cluster (i.e. one
            # of the two real TEAMS) as a referee candidate instead, and
            # accepted it, because that team's distance to the *outlier*
            # (not to the other team) exceeded the ratio -- the outlier
            # being an unrepresentative near-neighbor of one team is not
            # evidence the other team is a referee. A real team should
            # never be tested as a referee candidate at all. The accepted
            # candidate must ALSO clear color distinctiveness -- distance
            # to the nearer of the other two clusters must be at least
            # REFEREE_MIN_DISTINCTIVENESS_RATIO times those other two
            # clusters' own mutual distance, since a real referee kit is
            # chosen specifically to not resemble either team's, while a
            # same-team outlier sits closer to both team clusters than
            # they sit to each other.
            by_size = sorted(range(3), key=lambda k: counts[k])
            referee_cluster = None
            for k in by_size:
                cluster_patterned_frac = trusted_patterned[labels_trusted == k].mean() if counts[k] else 1.0
                if cluster_patterned_frac >= 0.5:
                    continue  # likely a patterned team's own small fragment -- try the next-smallest cluster
                other = [j for j in range(3) if j != k]
                d_other = float(np.linalg.norm(centroids_trusted[other[0]] - centroids_trusted[other[1]]))
                d_cand = min(
                    float(np.linalg.norm(centroids_trusted[k] - centroids_trusted[other[0]])),
                    float(np.linalg.norm(centroids_trusted[k] - centroids_trusted[other[1]])),
                )
                if (
                    d_other >= self.MIN_TEAM_SEPARATION_FOR_REFEREE_CHECK
                    and d_cand / d_other >= self.REFEREE_MIN_DISTINCTIVENESS_RATIO
                ):
                    referee_cluster = k
                break  # stop after the first non-patterned candidate either way -- never test a larger cluster
            if referee_cluster is not None:
                team_clusters = [k for k in range(3) if k != referee_cluster]
                self.team_centroids = np.array([centroids_trusted[team_clusters[0]], centroids_trusted[team_clusters[1]]])
                self.referee_centroid = centroids_trusted[referee_cluster]
            else:
                # No cluster confidently reads as a real referee this frame
                # (the common case when the referee isn't even in the
                # bootstrap frame) -- bootstrap team_a/team_b from the two
                # LARGER of the 3 k=3 clusters directly (already computed
                # above), deliberately EXCLUDING the smallest/outlier
                # cluster's sample(s) from team bootstrap entirely, rather
                # than re-clustering everyone with a fresh k=2 fit. This
                # matters: a fresh k=2 fit would be forced to fold the
                # ambiguous outlier into one of only two clusters, which
                # was confirmed to reintroduce this exact bug one level
                # removed -- the contaminated resulting centroid then made
                # OTHER genuine same-team samples look spuriously
                # "distinct" against it later. Treating the ambiguous
                # sample as simply uncertain (in neither team nor referee)
                # for this one bootstrap frame is consistent with this
                # class's existing abstain-rather-than-guess philosophy.
                # Referee detection is retried on later frames below,
                # using this clean, uncontaminated team baseline.
                by_size_desc = list(reversed(by_size))
                self.team_centroids = np.array([centroids_trusted[by_size_desc[0]], centroids_trusted[by_size_desc[1]]])

        if self.referee_centroid is None:
            # Retry referee identification on every frame until some frame
            # naturally contains a confidently color-distinct sample (or
            # never does for the whole clip -- an honest lack of any
            # `referee` label is preferred over a permanent wrong guess).
            trust_mask = ~occluded & ~patterned_arr
            d_teams = float(np.linalg.norm(self.team_centroids[0] - self.team_centroids[1]))
            if np.any(trust_mask) and d_teams >= self.MIN_TEAM_SEPARATION_FOR_DEFERRED_CHECK:
                cand_x = x[trust_mask]
                d_a = np.linalg.norm(cand_x - self.team_centroids[0], axis=1)
                d_b = np.linalg.norm(cand_x - self.team_centroids[1], axis=1)
                nearer = np.minimum(d_a, d_b)
                is_referee_like = nearer >= self.REFEREE_MIN_DISTINCTIVENESS_RATIO * d_teams
                if np.any(is_referee_like):
                    self.referee_centroid = cand_x[is_referee_like].mean(axis=0)

        centroids = (
            np.vstack([self.team_centroids, self.referee_centroid[None, :]])
            if self.referee_centroid is not None else self.team_centroids
        )
        n_clusters = centroids.shape[0]

        # Re-derived from the (possibly just-updated) centroids either way,
        # so occluded/bootstrap samples get a distance-based label/None
        # consistently with every later call, not a first-frame special case.
        dists = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
        d_sorted = np.sort(dists, axis=1)
        confident = d_sorted[:, 1] >= self.min_separation_ratio * np.maximum(d_sorted[:, 0], 1e-6)
        trust_sample = confident & ~occluded
        nearest = dists.argmin(axis=1)

        if n_clusters == 3:
            # Hard veto: a patterned sample nearest to the referee centroid
            # is reassigned to whichever TEAM centroid (0/1) is actually
            # closer -- it must never be allowed to read as referee, per
            # this function's own docstring.
            vetoed = patterned_arr & (nearest == 2)
            if np.any(vetoed):
                nearest[vetoed] = dists[vetoed][:, :2].argmin(axis=1)

        for k in range(n_clusters):
            assigned = x[trust_sample & (nearest == k)]
            if len(assigned):
                centroids[k] = (1 - self.ema_alpha) * centroids[k] + self.ema_alpha * assigned.mean(axis=0)
        self.team_centroids = centroids[:2]
        if n_clusters == 3:
            self.referee_centroid = centroids[2]

        return [int(nearest[i]) if trust_sample[i] else None for i in range(n)]
