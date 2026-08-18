"""Per-frame pitch calibration via a pretrained keypoint-detection model
(2026-08-11, roboflow/sports's pitch-keypoint model -- see PROGRESS.md's
2026-08-11 entry for the investigation this resolves).

`pitch_calibration_cv.py`'s classical-CV calibration is computed ONCE per
shot (on the shot's first frame) and reused for that shot's entire
duration -- a design that silently assumes an approximately-static camera
within a shot. That held for this project's short, cut-heavy SoccerNet
clips (each shot a few seconds), but was confirmed FALSE for a long,
continuously-panning broadcast shot (`final_mundial/jugada_clip.mp4`'s
single ~31.5s shot): the resulting pitch-line overlay in `render_demo.py`
visibly drifted completely off the real pitch markings within ~15s of the
shot's start, confirmed with direct frame evidence.

Recomputing calibration fresh from each frame's OWN detected pitch
landmarks (32 named keypoints -- line intersections, box corners, the
center circle) fixes this structurally: it never assumes a previous
frame's fit still applies, so it naturally tracks camera movement.
Confirmed directly: the same frames that showed severe drift under the
per-shot approach show a pitch-line overlay correctly locked onto the
real center circle, penalty boxes, and touchlines when recalibrated this
way.

This module is currently used only by `render_demo.py`'s pitch-line
overlay -- it does NOT replace `pitch_calibration_cv.py` for the real
per-row world positions the rest of the pipeline (Layers 2-4) depends on;
that is a larger, separate change (see the roboflow/sports scoping in
PROGRESS.md) not made here.

WEIGHTS_PATH points to the downloaded .pt file (gitignored, not committed
-- same convention as the other Roboflow-sourced weights in this
project). `calibrate_frame` returns `None` gracefully if the weights
aren't present, or if too few confident keypoints are found in a given
frame to fit a homography -- callers should skip drawing/using calibration
for that frame rather than trust a degenerate fit."""
from __future__ import annotations

from pathlib import Path

import numpy as np

WEIGHTS_PATH = Path("weights/roboflow_pitch_keypoint_detector.pt")
MIN_CONF = 0.5
# 5, not 4 (2026-08-12): the leave-one-out plausibility check (see
# MAX_LOO_RESIDUAL_FRAC's comment) needs at least 4 points remaining after
# holding one out, so it requires 5 confident keypoints to run at all. A
# bare 4-point fit is exactly the minimum cv2.findHomography can compute
# and can't be cross-validated against itself -- reject it outright rather
# than trust an unvalidatable fit.
MIN_KEYPOINTS_FOR_HOMOGRAPHY = 5

# Plausibility gate (2026-08-11): a real regression found during
# validation, not assumed -- on a Leicester-Man City frame, 8 confident
# keypoints were found (well above MIN_KEYPOINTS_FOR_HOMOGRAPHY), but 7 of
# the 8 were clustered in the left third of the frame with only 1 far to
# the right acting as an unstable "lever" point. `cv2.findHomography`
# still returns a numerically valid matrix for this configuration -- with
# so few effectively-independent constraints, the fit satisfies all of its
# own input points almost exactly, so a naive residual-on-fit-points check
# would not catch it either. The first approach tried here mirrored
# pitch_calibration_cv.py's own `_is_plausible_calibration` (reject a
# homography whose own frame corners map outside a padded pitch rect) --
# confirmed NOT to work for this keypoint-based case: it both let the
# Leicester failure through (a systematically-shifted-but-not-huge span
# can still numerically overlap the padded pitch rect) and rejected
# legitimately good tight-framing shots (final_mundial t=0/t=15, where the
# frame's own corners genuinely show crowd/stands, i.e. correctly
# off-pitch, even though the fit is accurate near the real detected
# keypoints). Bounding-box/convex-hull spread of the input keypoints was
# tried next and also failed to discriminate (final_mundial's confirmed
# good t=0 case has a SMALLER keypoint bbox area fraction than Leicester's
# bad case).
#
# What actually discriminates, confirmed directly: leave-one-out
# cross-validation. Refit the homography with each keypoint held out in
# turn and measure how far the fit's prediction lands from that point's
# real detected pixel position. A well-distributed keypoint set predicts
# each held-out point accurately (median ~1% of frame diagonal on all
# confirmed-good final_mundial frames, max <= 4.2%). Leicester's
# clustered-plus-lever-point set predicts its lever point terribly when
# that point is held out (14.5% of frame diagonal) precisely because nothing
# else in the cluster constrains that region of the fit -- the instability
# that makes the fit untrustworthy for extrapolation shows up directly as
# LOO error, unlike a residual computed on the fit's own training points.
MAX_LOO_RESIDUAL_FRAC = 0.08

# Second, complementary plausibility signal (2026-08-12): LOO alone was
# confirmed insufficient during regression-testing against Swansea-Man
# Utd's clip -- two separate tight referee/player close-ups (t=5s, t=8s,
# no real pitch markings visible at all, just grass and a person) each
# produced several confident keypoints, clustered in one small region of
# the frame, that were still mutually self-consistent enough to pass
# MAX_LOO_RESIDUAL_FRAC's bar -- LOO only measures internal consistency of
# the point set, not whether the fit is plausible at all. What both cases'
# homographies actually imply is physically absurd: a handful of points
# crammed into a third of the frame's diagonal are claimed to represent
# tens of meters of real pitch -- an implied scale (cm of real pitch per
# detected pixel) well above every confirmed-good case checked directly
# by rendering and visually inspecting the resulting overlay (4.78-9.62
# cm/px across final_mundial, Chelsea-Burnley, Arsenal-Anderlecht,
# Hull-Arsenal), while Swansea's two bad fits imply 10.82 and 12.16 cm/px.
# This ratio is NOT a stable, resolution-independent, zoom-invariant
# absolute threshold across an entire clip on its own (confirmed directly:
# a legitimately good Chelsea-Burnley frame at 11.38 cm/px sits ABOVE one
# of Swansea's confirmed-bad frames at 10.82 cm/px -- no single cm/px
# threshold perfectly separates every case found so far). Given that
# overlap, and this module's own stated design principle (skip drawing
# rather than trust a degenerate fit), the threshold below is set
# conservatively just under both confirmed-bad Swansea values rather than
# threading the narrow gap precisely -- it reliably rejects every
# confirmed-bad case with a comfortable margin, at the disclosed cost of
# also rejecting a small number of legitimate borderline-zoom frames
# (e.g. one otherwise-plausible Chelsea-Burnley frame at 11.38 cm/px).
MAX_WORLD_CM_PER_PX = 10.0

_model = None
_checked = False
_world_points = None
_pitch_config = None


def _get_model():
    global _model, _checked, _world_points, _pitch_config
    if _checked:
        return _model
    _checked = True
    if not WEIGHTS_PATH.exists():
        return None
    from ultralytics import YOLO
    from sports.configs.soccer import SoccerPitchConfiguration
    print("Loading pitch-keypoint calibration model (first call only)...")
    _model = YOLO(str(WEIGHTS_PATH))
    _pitch_config = SoccerPitchConfiguration()
    _world_points = np.array(_pitch_config.vertices, dtype=np.float32)
    print("Loaded pitch-keypoint calibration model.")
    return _model


def pitch_config():
    """The sports package's own pitch layout (vertices in cm on a 120x70m
    pitch, edges connecting them, center-circle radius) -- exposed so
    callers can draw the same real markings this module calibrates
    against, without hand-rolling a second, differently-scaled set of
    pitch coordinates (this project's own PITCH_LENGTH_M/PITCH_WIDTH_M
    elsewhere assume a 105x68m pitch -- a different convention, not
    interchangeable with this one)."""
    _get_model()
    return _pitch_config


def _is_plausible(src_px: np.ndarray, tgt_world: np.ndarray, frame_w: int, frame_h: int) -> bool:
    """Two independent checks, both required to pass -- see
    MAX_LOO_RESIDUAL_FRAC's and MAX_WORLD_CM_PER_PX's comments for why
    neither alone is sufficient."""
    import cv2
    n = len(src_px)
    diag = float(np.hypot(frame_w, frame_h))

    px_diag = float(np.hypot(
        src_px[:, 0].max() - src_px[:, 0].min(),
        src_px[:, 1].max() - src_px[:, 1].min(),
    ))
    world_diag = float(np.hypot(
        tgt_world[:, 0].max() - tgt_world[:, 0].min(),
        tgt_world[:, 1].max() - tgt_world[:, 1].min(),
    ))
    if px_diag < 1.0 or world_diag / px_diag > MAX_WORLD_CM_PER_PX:
        return False

    for i in range(n):
        idx = [j for j in range(n) if j != i]
        h_loo, _ = cv2.findHomography(tgt_world[idx], src_px[idx], 0)
        if h_loo is None:
            return False
        pred = cv2.perspectiveTransform(tgt_world[i].reshape(1, 1, 2), h_loo).reshape(2)
        if np.linalg.norm(pred - src_px[i]) / diag > MAX_LOO_RESIDUAL_FRAC:
            return False
    return True


# Edge-support radius (2026-08-17): a real regression found from direct
# video review, not assumed -- on a final_mundial frame (t=8s) with 8
# confident keypoints, all clustered near one goal box (world x=0-2015cm),
# the fit passed both plausibility checks above (LOO_max=0.043, well under
# 0.08; cm/px=4.78, well under 10.0) and still produced a visibly wrong
# overlay: the far box/touchline, whose nearest matched keypoint is
# 80-100m away, got drawn via wild extrapolation, cutting a nonsense
# diagonal line across empty grass, while the near-box lines (matching
# the actual keypoints) were precisely correct. LOO and the cm/px check
# both validate the FIT as a whole; neither says anything about how far
# any individual drawn line is from a point the fit actually has support
# near. Confirmed directly this isn't just a "draw less" tradeoff: a
# genuinely wide, GOOD frame (final_mundial t=15, confirmed excellent by
# direct visual inspection) has real matched keypoints at the halfway
# line and the center circle's own edge (world x=6000cm), so those
# regions stay eligible under a support-radius filter, while its own far
# box/touchline (x=9985-12000cm, nothing matched anywhere near there
# either) gets correctly excluded too -- previously drawn on luck/being
# off-screen, not on any real support. 2000cm (20m, roughly a penalty
# box's own depth) sits comfortably above the near-box edges' own
# distances in the bad t=8 case (275-1008cm) and comfortably below the
# next tier out (the halfway line at ~3988cm in that same bad frame).
EDGE_SUPPORT_RADIUS_CM = 2000.0


def points_near_support(query_world_pts: np.ndarray, matched_world_pts: np.ndarray) -> np.ndarray:
    """Boolean mask, one entry per row of `query_world_pts` -- True for
    any world point within EDGE_SUPPORT_RADIUS_CM of at least one of this
    frame's own matched keypoints (`matched_world_pts`, as returned by
    `calibrate_frame`). Generic over whatever points a caller wants
    checked (named pitch vertices, arbitrary circle sample points, ...) --
    see EDGE_SUPPORT_RADIUS_CM's comment for why extrapolating past it
    isn't trustworthy even when the fit itself passes the plausibility
    gate."""
    dists = np.linalg.norm(
        query_world_pts[:, None, :].astype(np.float64)
        - matched_world_pts[None, :, :].astype(np.float64), axis=2
    )
    return dists.min(axis=1) <= EDGE_SUPPORT_RADIUS_CM


def vertices_near_support(matched_world_pts: np.ndarray) -> np.ndarray:
    """Boolean mask over `pitch_config().vertices` -- convenience wrapper
    around `points_near_support` for the 32 named pitch vertices
    specifically (what `render_demo.py`'s line-edge drawing uses).
    Callers should only draw a line between two vertices that both pass
    this check."""
    cfg = pitch_config()
    all_vertices = np.array(cfg.vertices, dtype=np.float64)
    return points_near_support(all_vertices, matched_world_pts)


def calibrate_frame(frame_bgr: np.ndarray):
    """Returns `(transformer, matched_world_pts)` -- `transformer` is a
    `sports.common.view.ViewTransformer` mapping this module's world
    pitch coordinates (cm, see `pitch_config()`) to this specific frame's
    pixel coordinates; `matched_world_pts` is the set of real-world
    positions this frame's own confident keypoints were matched to (see
    `vertices_near_support`, which callers should use before drawing any
    line far from them). Returns `None` (not a tuple) if the model isn't
    available, too few confident keypoints were found in this frame, or
    the resulting homography fails the plausibility gate (see
    MAX_LOO_RESIDUAL_FRAC's comment)."""
    model = _get_model()
    if model is None:
        return None
    from sports.common.view import ViewTransformer
    results = model(frame_bgr, verbose=False)[0]
    if results.keypoints is None or len(results.keypoints.xy) == 0:
        return None
    kpts_xy = results.keypoints.xy[0].cpu().numpy()
    kpts_conf = (
        results.keypoints.conf[0].cpu().numpy() if results.keypoints.conf is not None
        else np.ones(len(kpts_xy))
    )
    mask = kpts_conf > MIN_CONF
    if mask.sum() < MIN_KEYPOINTS_FOR_HOMOGRAPHY:
        return None
    src_px = kpts_xy[mask].astype(np.float32)
    tgt_world = _world_points[mask]
    h, w = frame_bgr.shape[:2]
    if not _is_plausible(src_px, tgt_world, w, h):
        return None
    try:
        transformer = ViewTransformer(source=tgt_world, target=src_px)
    except ValueError:
        return None
    return transformer, tgt_world
