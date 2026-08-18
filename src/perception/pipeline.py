"""Layer 1 entry point: reads a video, runs detection + team ID + tracking +
calibration per frame, and returns a per-frame per-object position table --
the single artifact every downstream layer consumes. See the project spec section 4.

Backend selection: real broadcast footage should use YOLOv8 (`backend="yolo"`);
the synthetic test clip uses the HSV color detector (`backend="color"`) since
pretrained COCO YOLO does not recognize painted circles as people/balls. Both
backends produce the same output schema so Layers 2-4 don't care which ran.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from src.perception import (
    color_detector, kit_pattern_classifier, pitch_calibration_cv, player_classifier, pose_estimator,
    roboflow_referee_detector, scene_cut, team_id, yolo_detector,
)
from src.perception.bytetrack_lite import ByteTrackLite
from src.perception.calibration import PitchCalibrator

_SHOT_TRACK_ID_STRIDE = 100_000  # keeps track_ids globally unique across shots

# Crowd/sideline filter (2026-07-17): a "person" detection whose calibrated
# real-world position falls well outside the actual pitch is not a player at
# all -- it's a spectator, bench/technical-area personnel, or a ball boy
# that the detector can't otherwise tell apart from a player. Reuses the
# calibration homography already computed for every row rather than a new
# model. Margin is generous enough to keep the technical area/dugouts
# (players warming up, staff) without accepting the stands, which in a real
# broadcast's projected pitch coordinates sit well beyond this.
SIDELINE_MARGIN_M = 5.0

# Goalkeeper-as-referee correction (2026-07-18): TeamColorAnchor's
# population-size referee heuristic (team_id.py) has a real blind spot --
# a goalkeeper's kit is ALSO deliberately distinct from both outfield
# teams' (IFAB Law 4), so it can just as easily be the smallest bootstrap
# color cluster as the real referee is. Confirmed directly on
# foul_leicester_mancity.mp4: track 13, labeled "referee", is visibly the
# Man City goalkeeper standing in his own goal mouth (white kit) -- the
# real referee (cyan kit) is a separate, correctly-labeled detection
# nearby. Color alone can't distinguish the two (both are legitimately
# singleton, distinct-color clusters at bootstrap time) -- but POSITION
# can: a real referee ranges across the whole pitch following play, while
# a goalkeeper spends the large majority of their time within their own
# defensive zone. Confirmed empirically on the same clip: the real
# goalkeeper tracks (13, 57) stay within 25m of one goal line for
# 100%/~75%+ of their own-calibration rows respectively; the two genuine
# referee tracks (12, 59) never come within 35m of either goal line at
# all. GK_MIN_ROWS avoids reclassifying a short, noisy track from a
# handful of frames (e.g. a referee standing still near a goal for one
# set piece).
GK_ZONE_M = 25.0
GK_ZONE_MIN_FRACTION = 0.7
GK_MIN_ROWS = 15

# Scale gate for the Roboflow referee/goalkeeper detector (2026-07-20): real
# testing (see PROGRESS.md) found this second detector reliably fails --
# missing real players entirely, or finding only tiny fragments -- once a
# broadcast zooms in close (tackles, tangles), across every clip tested.
# Root-caused to a shot-scale effect, not crowding: mean person-box area as
# a fraction of frame area separated every "worked" frame (0.15-0.38%) from
# every "failed" frame (3.8-16.8%) with a clean, zero-overlap gap. This
# threshold sits in that gap -- comfortably below the failing range, so a
# frame is only ever handed to the second detector when it's confidently a
# wide/tactical shot, the framing that model was actually validated on.
PERSON_BOX_MEAN_AREA_FRAC_THRESHOLD = 0.01

# Box-quality gate for accepting a Roboflow match (2026-07-21): validation
# on Arsenal-Anderlecht found a real `is_goalkeeper` false positive -- a
# tiny (35x60px), low-quality sideline/crowd sliver, tracked pinned to the
# frame's left edge (x1 in [0.0, 4.6] for its ENTIRE lifetime) got
# IoU-matched to a Roboflow "goalkeeper" detection that was itself
# genuinely confident (0.88) -- Roboflow's own confidence can't catch
# this, since it's confidently reading a REAL goalkeeper-shaped person
# elsewhere; the primary detector's box is what's unreliable here. Checked
# directly against every confirmed-correct match this project has found
# (Chelsea-Burnley's referee, conf 0.65-0.77, no edge contact; Leicester's
# goalkeeper, conf 0.55-0.75, no edge contact): both comfortably clear a
# minimum-confidence bar AND never touch a frame boundary, while the bad
# case fails the edge check specifically (confidence alone overlaps too
# much with genuine cases, 0.46-0.73, to be a clean discriminator by
# itself). A primary-detector box this pinned to a frame edge is a classic
# partial/clipped-detection signal (a person, sign, or object straddling
# the frame boundary), independent of whatever Roboflow thinks is there.
# Does NOT address every known Roboflow error -- the Anderlecht purple-kit
# false referee (2026-07-20) is a confident, well-formed, WRONG call by
# Roboflow itself (checked directly, conf 0.79, no edge contact) that no
# primary-box-quality check can catch; that remains a disclosed, open
# limitation of the model, not something this gate is meant to fix.
ROBOFLOW_MATCH_MIN_PRIMARY_CONF = 0.4
ROBOFLOW_MATCH_FRAME_EDGE_MARGIN_PX = 2.0

# Box-size anomaly guard for the appearance classifier (2026-07-19): a
# row's box height must fall within this multiple of that same track's
# own median box height elsewhere in the clip for the appearance check to
# be trusted at all -- see `_run_pose_pass2`'s docstring for the concrete
# case (a real player's box corrupted/oversized during a violent tackle,
# fed to the classifier, confidently misread as crowd).
BOX_SIZE_ANOMALY_RATIO = 1.6


def _reclassify_goalkeepers(df: pd.DataFrame) -> pd.DataFrame:
    """Re-labels any `"referee"`-cls track that is positionally a
    goalkeeper (see the constants above) back to `"player"`, on whichever
    team's goal it's confined near, and tags it `is_goalkeeper=True` --
    a real, usable signal for downstream consumers rather than a guess
    (e.g. handball legality, an already-documented open gap: see
    pose_signals.py's handball docstring, "goalkeepers... are not
    distinguishable from outfield players yet"). Only uses
    `calib_source == "own"` rows for the positional check, so an
    unreliable position doesn't drive a reclassification decision.

    2026-07-20: `is_goalkeeper` may already be `True` on some rows here --
    the Roboflow referee/goalkeeper detector (see `roboflow_referee_detector.py`)
    can confirm a goalkeeper directly on wide-shot frames, bypassing this
    positional heuristic entirely for those rows. Only initializes the
    column where it doesn't already exist, so those pre-set values survive;
    this function still runs its own positional check afterward for
    whatever that second detector didn't catch (e.g. close/zoomed frames
    its scale gate excluded it from).

    2026-08-17: also REVOKES `is_goalkeeper` from any track that fails
    this same positional check, regardless of which mechanism set it --
    a real, disclosed gap found on Arsenal-Anderlecht (see PROGRESS.md):
    Roboflow's own "goalkeeper" class confidently (and wrongly) tags
    ordinary, centrally-positioned field players, and until now nothing
    ever re-checked that tag against position at all, unlike the
    referee-promotion path above which always required it. Two other
    signals were tested directly first and found NOT to discriminate this
    class of false positive: the appearance-based player/non-player
    classifier (`player_classifier.py`) reads a confirmed non-player
    (a pitch-side photographer, final_mundial track 315) as player-like in
    8/8 sampled crops, and per-track speed statistics didn't cleanly
    separate a confirmed real goalkeeper from that same confirmed
    non-player. Position, by contrast, is exactly what the referee-
    promotion path above was already built and validated against real
    goalkeeper tracks for -- reused here unchanged (same constants, same
    calib_source=="own"-only reliability requirement), not a new,
    unvalidated threshold. A track with fewer than GK_MIN_ROWS reliable
    rows is revoked too (not given the benefit of the doubt): the
    promotion path above requires that same minimum to ever assert
    is_goalkeeper=True in the first place, so a tag that can't clear the
    same bar under revocation shouldn't have been trusted either.

    Confirmed directly this does NOT address the final_mundial photographer
    case (track 315) -- checked, not assumed: his own calibrated position
    DOES satisfy this exact zone check (x within 25m of a goal line for
    100% of his reliable rows), the same way a real goalkeeper's would, so
    he survives this revocation pass unchanged. That case remains open --
    see PROGRESS.md for the full record of what was tried (position,
    appearance, speed) and why none of them discriminate a goal-line
    photographer from a real goalkeeper positionally confined the same
    way."""
    if "is_goalkeeper" not in df.columns:
        df["is_goalkeeper"] = False
    if "cls" not in df.columns or df.empty:
        return df
    pitch_length = pitch_calibration_cv.PITCH_LENGTH_M
    for track_id in df.loc[df["cls"] == "referee", "track_id"].unique():
        mask = df["track_id"] == track_id
        reliable = df.loc[mask & (df["calib_source"] == "own")]
        if len(reliable) < GK_MIN_ROWS:
            continue
        in_zone = (reliable["x"] <= GK_ZONE_M) | (reliable["x"] >= pitch_length - GK_ZONE_M)
        if in_zone.mean() < GK_ZONE_MIN_FRACTION:
            continue
        team = "team_a" if reliable["x"].mean() < pitch_length / 2 else "team_b"
        df.loc[mask, "cls"] = "player"
        df.loc[mask, "team"] = team
        df.loc[mask, "is_goalkeeper"] = True

    for track_id in df.loc[df["is_goalkeeper"], "track_id"].unique():
        mask = df["track_id"] == track_id
        reliable = df.loc[mask & (df["calib_source"] == "own")]
        if len(reliable) < GK_MIN_ROWS:
            df.loc[mask, "is_goalkeeper"] = False
            continue
        in_zone = (reliable["x"] <= GK_ZONE_M) | (reliable["x"] >= pitch_length - GK_ZONE_M)
        if in_zone.mean() < GK_ZONE_MIN_FRACTION:
            df.loc[mask, "is_goalkeeper"] = False
    return df


def _run_color_backend(video_path: str, calibrator: PitchCalibrator) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    tracker = ByteTrackLite()
    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = color_detector.detect_frame(frame, frame_idx)
        det_dicts = [
            {"cls": d.cls, "team": d.team_hint, "box": (d.x1, d.y1, d.x2, d.y2), "conf": d.conf}
            for d in dets
        ]
        tracked = tracker.update(det_dicts)
        for t in tracked:
            cx, cy = (t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2
            x_m, y_m = calibrator.pixel_to_pitch(cx, cy)
            rows.append({
                "frame": frame_idx, "time_s": frame_idx / fps, "track_id": t["track_id"],
                "cls": t["cls"], "team": t["team"], "x": x_m, "y": y_m, "conf": t["conf"],
            })
        frame_idx += 1
    cap.release()
    df = pd.DataFrame(rows)
    df["is_goalkeeper"] = False  # schema symmetry with the yolo backend; not modeled in the synthetic clip
    return df


def _run_yolo_backend_shot(video_path: str, calibrator: PitchCalibrator, fps: float,
                            start_frame: int, end_frame: int, track_id_offset: int,
                            team_anchor: team_id.TeamColorAnchor, calib_source: str,
                            processed_so_far: int, total_frames: int) -> tuple[list[dict], int]:
    """Pass 1 of the two-pass (VAR-style) architecture (2026-07-18, see
    reports/two_pass_architecture_scoping.md): detection + team ID +
    tracking over a single shot's frame range, deliberately WITHOUT pose
    estimation -- that's the expensive half of what this function used to
    do unconditionally on every frame, and it now only runs inside the
    short review windows Pass 1's own output flags (`_run_pose_pass2`,
    called once per clip after every shot's Pass-1 rows are assembled).

    A fresh tracker is used per shot -- track identity across a cut is
    meaningless (it's a different framing, possibly a different part of
    the pitch or a different subject entirely), so continuing the same
    tracker across a cut would silently associate unrelated detections.

    Team identity (`team_anchor`) is the opposite: it's passed in from the
    caller and shared across every shot in the clip, not recreated here --
    see TeamColorAnchor's docstring for why re-clustering blind per shot
    (or per frame) is the bug it fixes.

    `processed_so_far`/`total_frames` are only for the periodic progress
    print -- YOLO detection per frame is the slow part of this pipeline
    (visible as a long silent pause otherwise), so this prints every 100
    frames across the whole clip, not just this shot."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    tracker = ByteTrackLite()
    rows = []
    processed = processed_so_far
    for frame_idx in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        boxes = yolo_detector.detect_frame(frame, frame_idx)
        torso_colors, person_box_tuples, person_confs = [], [], []
        for b in boxes:
            if b.cls == "person":
                torso_colors.append(team_id.torso_crop_mean_color(frame, b.x1, b.y1, b.x2, b.y2))
                person_box_tuples.append((b.x1, b.y1, b.x2, b.y2))
                person_confs.append(b.conf)

        # Second opinion from the Roboflow referee/goalkeeper detector
        # (2026-07-20) -- only on frames confidently wide enough for it to
        # be reliable (see PERSON_BOX_MEAN_AREA_FRAC_THRESHOLD's comment).
        # Box EXISTENCE always stays with the primary detector above; this
        # only ever relabels boxes that already exist.
        if person_box_tuples:
            frame_area = frame.shape[0] * frame.shape[1]
            mean_area_frac = sum(
                (x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in person_box_tuples
            ) / len(person_box_tuples) / frame_area
        else:
            mean_area_frac = 0.0
        if person_box_tuples and mean_area_frac < PERSON_BOX_MEAN_AREA_FRAC_THRESHOLD:
            roboflow_labels = roboflow_referee_detector.classify_boxes(frame, person_box_tuples)
        else:
            roboflow_labels = [None] * len(person_box_tuples)

        # Box-quality gate (2026-07-21, see ROBOFLOW_MATCH_MIN_PRIMARY_CONF's
        # comment): a Roboflow match is only trusted for a box the PRIMARY
        # detector itself is reasonably confident about and that isn't
        # pinned to a frame edge (a classic partial/clipped-detection
        # signal) -- Roboflow's own confidence can't catch this, since it
        # can be genuinely confident about a real person elsewhere while
        # the primary box it got IoU-matched to is the unreliable one.
        frame_h, frame_w = frame.shape[0], frame.shape[1]
        m = ROBOFLOW_MATCH_FRAME_EDGE_MARGIN_PX

        def _box_quality_ok(box, conf):
            x1, y1, x2, y2 = box
            if conf < ROBOFLOW_MATCH_MIN_PRIMARY_CONF:
                return False
            return not (x1 <= m or y1 <= m or x2 >= frame_w - m or y2 >= frame_h - m)

        roboflow_labels = [
            rf if (rf is None or _box_quality_ok(box, conf)) else None
            for rf, box, conf in zip(roboflow_labels, person_box_tuples, person_confs)
        ]
        referee_confirmed = [rf == "referee" for rf in roboflow_labels]
        goalkeeper_confirmed = [rf == "goalkeeper" for rf in roboflow_labels]
        non_referee_confirmed = [rf in ("player", "goalkeeper") for rf in roboflow_labels]

        # Hard domain constraint (2026-07-19): a patterned/striped kit
        # crop can never be a real referee (IFAB Law 4) -- see
        # kit_pattern_classifier.py and TeamColorAnchor.assign's own
        # docstring for why this is a trained classifier, not a
        # hand-crafted color statistic.
        patterned_flags = kit_pattern_classifier.classify_boxes(frame, person_box_tuples)
        team_labels = team_anchor.assign(torso_colors, person_box_tuples, patterned_flags) if torso_colors else []

        # Roboflow override (2026-07-20): applied strictly AFTER
        # TeamColorAnchor's own call above, as a pure post-hoc relabel --
        # never changes what TeamColorAnchor itself sees or learns from,
        # so its own bootstrap/EMA centroid evolution is completely
        # undisturbed. This is the same safe pattern already proven for
        # the appearance classifier in Pass 2 (see `_run_pose_pass2`'s
        # docstring). A first version of this instead EXCLUDED
        # Roboflow-confirmed-referee samples from TeamColorAnchor's input
        # entirely -- validation on Leicester-Man City found this starved
        # TeamColorAnchor's own referee-detection logic (built last
        # session) of exactly the clean samples it needs, causing it to
        # learn a WORSE centroid from noisier frames instead and
        # misclassify far more real players as referee on frames Roboflow
        # doesn't cover (referee rows 49->319, and the clip's 3
        # previously-validated real foul catches disappeared as collateral
        # damage). Never repeat that -- always let TeamColorAnchor process
        # every sample unfiltered, then override only the OUTPUT label.
        for i in range(len(person_box_tuples)):
            if referee_confirmed[i]:
                team_labels[i] = 2
            elif non_referee_confirmed[i] and team_labels[i] == 2:
                # TeamColorAnchor guessed referee but Roboflow confidently
                # says otherwise -- reassign to whichever team's centroid
                # is nearer, mirroring TeamColorAnchor's own patterned-veto
                # reassignment logic.
                if team_anchor.team_centroids is not None:
                    c = torso_colors[i]
                    d0 = float(np.linalg.norm(c - team_anchor.team_centroids[0]))
                    d1 = float(np.linalg.norm(c - team_anchor.team_centroids[1]))
                    team_labels[i] = 0 if d0 <= d1 else 1
                else:
                    team_labels[i] = None

        det_dicts = []
        person_i = 0
        for b in boxes:
            if b.cls == "person":
                label = team_labels[person_i]
                is_gk = goalkeeper_confirmed[person_i]
                person_i += 1
                if label == 2:
                    # Referee (see TeamColorAnchor): a distinct tracker
                    # class, not "person"/team_a/team_b, so it's excluded
                    # from every downstream player-only signal by
                    # construction rather than by an extra filter.
                    det_cls, team = "referee", None
                else:
                    det_cls = "person"
                    team = None if label is None else f"team_{'a' if label == 0 else 'b'}"
            else:
                det_cls, team, is_gk = b.cls, None, False
            det_dicts.append({
                "cls": det_cls, "team": team, "box": (b.x1, b.y1, b.x2, b.y2), "conf": b.conf,
                "is_goalkeeper": is_gk,
            })

        tracked = tracker.update(det_dicts)
        for t in tracked:
            cx, cy = (t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2
            x_m, y_m = calibrator.pixel_to_pitch(cx, cy)
            cls = {"person": "player", "referee": "referee", "ball": "ball"}.get(t["cls"], t["cls"])
            # 2026-08-12: this used to only run for cls == "player" -- a real,
            # structural gap (see PROGRESS.md's 2026-08-11 final_mundial entry)
            # that let a crowd/photographer detection TeamColorAnchor's color
            # logic classified "referee" bypass this check entirely, since it
            # never even reached the branch below. Confirmed directly on that
            # clip: a fan in the stands (track 313, t~25.9s) calibrated to
            # y=-19 to -22.8 (nowhere near the real 0-68m pitch width) but
            # was never excluded, purely because it was labeled "referee"
            # first. Both classes now run the same position check.
            if cls in ("player", "referee"):
                if calib_source != "own":
                    # The pitch-boundary check below depends entirely on
                    # the calibrated (x_m, y_m) being meaningful -- on a
                    # shot using fallback_prev_shot (a differently-framed
                    # preceding shot's reused homography) or the flat
                    # placeholder, it isn't, so the check's answer would be
                    # confidently wrong rather than just unreliable at the
                    # margins (2026-07-17: confirmed directly -- crowd
                    # detections on exactly these shots kept reading as
                    # "player" since the whole frame maps inside the
                    # padded pitch rectangle). Mark honestly as low
                    # confidence instead of trusting a geometric check
                    # built on positions we already know aren't real here.
                    # Referee stays "referee", deliberately NOT "low_confidence"
                    # (2026-08-12): "low_confidence" rows are treated as
                    # candidate players by pose_signals.py/review_windows.py's
                    # contact-detection logic -- correct for a player whose
                    # calibrated position just can't be trusted, but wrong for
                    # something already identified as the referee, which must
                    # stay excluded from player-only signals by construction
                    # (see the referee-assignment comment above) regardless of
                    # calibration quality.
                    if cls == "player":
                        cls = "low_confidence"
                elif not (
                    -SIDELINE_MARGIN_M <= x_m <= pitch_calibration_cv.PITCH_LENGTH_M + SIDELINE_MARGIN_M
                    and -SIDELINE_MARGIN_M <= y_m <= pitch_calibration_cv.PITCH_WIDTH_M + SIDELINE_MARGIN_M
                ):
                    # Calibrated position falls well outside the real pitch --
                    # not a player or referee at all (see SIDELINE_MARGIN_M
                    # above). Reclassified rather than dropped, so this is
                    # auditable the same way calib_source already is.
                    cls = "non_player"
            row = {
                "frame": frame_idx, "time_s": frame_idx / fps, "track_id": t["track_id"] + track_id_offset,
                "cls": cls, "team": t["team"],
                "x": x_m, "y": y_m, "conf": t["conf"], "calib_source": calib_source,
                "box_x1": t["box"][0], "box_y1": t["box"][1], "box_x2": t["box"][2], "box_y2": t["box"][3],
                # Roboflow-confirmed goalkeeper (2026-07-20), when available --
                # see `_reclassify_goalkeepers`, which now preserves this
                # instead of resetting it, and still positionally promotes
                # anything this second detector didn't catch (e.g. close/
                # zoomed frames the scale gate excluded it from).
                "is_goalkeeper": t.get("is_goalkeeper", False),
            }
            # kp_<name>_x/_y/_c columns are added afterward, for every row,
            # by `_run_pose_pass2` -- NaN outside a Pass-1-flagged review
            # window, real values inside one. Not populated here at all
            # (Pass 1 never calls the pose model), unlike before this
            # session's two-pass split.
            rows.append(row)

        processed += 1
        if processed % 100 == 0 or processed == total_frames:
            pct = 100 * processed / total_frames if total_frames else 0
            print(f"  perception: {processed}/{total_frames} frames ({pct:.0f}%)")
    cap.release()
    return rows, processed


def _calibrate_shot_own(video_path: str, start_frame: int) -> PitchCalibrator | None:
    """Tries real keypoint-based calibration (pitch_calibration_cv.py) on a
    shot's first frame. Returns None if no pitch keypoints are confidently
    detected -- the caller decides the fallback (nearest preceding shot's
    calibration, or the flat placeholder as a last resort); see
    `_run_yolo_backend`."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    boxes = yolo_detector.detect_frame(frame, start_frame)
    person_boxes = [(b.x1, b.y1, b.x2, b.y2) for b in boxes if b.cls == "person"]
    return pitch_calibration_cv.calibrate_frame(frame, person_boxes)


def _run_pose_pass2(video_path: str, df: pd.DataFrame, windows: list[tuple[float, float]], fps: float) -> pd.DataFrame:
    """Pass 2 of the two-pass (VAR-style) architecture (2026-07-18, see
    reports/two_pass_architecture_scoping.md): re-opens the clip and runs
    the dual-pass pose estimator ONLY on frames inside `windows` (Pass 1's
    flagged review windows), instead of unconditionally on every frame --
    this is where the redesign's compute savings come from, since pose
    estimation was the expensive half of what Layer 1 used to do on every
    single frame regardless of whether anything worth reviewing happened
    there.

    Adds the kp_<joint>_x/_y/_c columns to the FULL dataframe (NaN
    everywhere outside a window) -- the exact schema the yolo backend
    always produced, so every existing downstream consumer (contact_types,
    torso-fall, handball, sprint/jump analytics) works completely
    unchanged: each of those already treats a missing/NaN keypoint as "no
    data for this row" (confirmed directly -- e.g. `_torso_fall_runs`
    checks `"kp_l_shoulder_x" not in df.columns`, `contact_type_events`
    checks `"kp_nose_x" not in players.columns`), so none of them need to
    know windowing happened at all. Reuses Pass 1's own tracked boxes for
    each frame (via `df`) rather than re-running the primary YOLO
    detector -- Pass 2 only ever ADDS pose data on top of Pass 1's
    detection/tracking/team-ID decisions, it never revises them.

    Also re-applies the appearance-based player/non-player classifier
    (`player_classifier.py`, built and validated 2026-07-18, previously
    reverted from the pipeline) inside these same windows -- STRICTLY as
    a post-hoc relabeling of `cls`/`team`, never by calling
    `TeamColorAnchor.assign()` again. This is what makes it safe this
    time: Pass 1 already ran its ENTIRE team-color bootstrap/EMA
    evolution for the whole clip before Pass 2 ever starts (the two
    passes are sequential, not interleaved), so there is no live
    `TeamColorAnchor` state left for Pass 2 to touch at all -- last
    session's regression (filtering which crops fed `team_anchor.assign`
    shifted its own centroid evolution and cost the Swansea flagship
    catch elsewhere in the clip) structurally cannot recur here, since
    this function never calls `assign` a second time.

    Box-size anomaly guard (2026-07-19): re-validation on Chelsea-Burnley
    found the appearance classifier confidently (P(player)=0.003) but
    WRONGLY excluded one of the two real flagship tackle participants --
    traced directly to that row's own box being corrupted/oversized
    during the chaotic tackle, so the crop was dominated by background
    crowd with only a sliver of the real player visible. This is the same
    "real contact corrupts box geometry" phenomenon already documented
    elsewhere in this project (speed, aspect ratio) manifesting through a
    new pathway -- not a classifier generalization failure, since the
    crop genuinely looks like crowd. No confidence threshold fixes a
    reading this confidently wrong, so instead: skip the appearance check
    entirely for a row whose box height is a sudden outlier vs. that same
    track's own median box height elsewhere in the clip (BOX_SIZE_ANOMALY_RATIO)
    -- "too corrupted to trust either way," the same abstain-rather-than-
    guess spirit already used throughout this project, rather than acting
    on a classification of a box that likely isn't showing what Layer 1
    thinks it's showing."""
    for name in pose_estimator.KEYPOINT_NAMES:
        df[f"kp_{name}_x"] = float("nan")
        df[f"kp_{name}_y"] = float("nan")
        df[f"kp_{name}_c"] = float("nan")
    if not windows or df.empty:
        return df

    track_median_height = (df["box_y2"] - df["box_y1"]).groupby(df["track_id"]).transform("median")
    df["_median_box_height"] = track_median_height

    cap = cv2.VideoCapture(video_path)
    for t_start, t_end in windows:
        f_start = max(0, int(round(t_start * fps)))
        f_end = int(round(t_end * fps)) + 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
        for frame_idx in range(f_start, f_end):
            ok, frame = cap.read()
            if not ok:
                break
            frame_rows = df[(df["frame"] == frame_idx) & (df["cls"] != "ball")]
            if frame_rows.empty:
                continue
            person_box_tuples = list(zip(
                frame_rows["box_x1"], frame_rows["box_y1"], frame_rows["box_x2"], frame_rows["box_y2"],
            ))
            pose_detections = pose_estimator.estimate_frame(frame)
            keypoints_per_person = pose_estimator.associate_keypoints(person_box_tuples, pose_detections)
            for row_index, kpts in zip(frame_rows.index, keypoints_per_person):
                if kpts is None:
                    continue
                for k_i, name in enumerate(pose_estimator.KEYPOINT_NAMES):
                    df.at[row_index, f"kp_{name}_x"] = float(kpts[k_i][0])
                    df.at[row_index, f"kp_{name}_y"] = float(kpts[k_i][1])
                    df.at[row_index, f"kp_{name}_c"] = float(kpts[k_i][2])

            appearance_ok = player_classifier.classify_boxes(frame, person_box_tuples)
            if appearance_ok is None:
                continue  # classifier not trained in this checkout -- no-op, same as before this change
            for row_index, ok_appearance, box in zip(frame_rows.index, appearance_ok, person_box_tuples):
                if ok_appearance or df.at[row_index, "cls"] not in ("player", "low_confidence"):
                    continue
                median_h = df.at[row_index, "_median_box_height"]
                box_h = box[3] - box[1]
                if pd.notna(median_h) and median_h > 0 and (
                    box_h > BOX_SIZE_ANOMALY_RATIO * median_h or box_h < median_h / BOX_SIZE_ANOMALY_RATIO
                ):
                    continue  # box is a size outlier for this track -- don't trust the crop either way
                df.at[row_index, "cls"] = "non_player"
                df.at[row_index, "team"] = None
                df.at[row_index, "is_goalkeeper"] = False
    cap.release()
    df.drop(columns=["_median_box_height"], inplace=True)
    return df


def _run_yolo_backend(video_path: str, frame_w: int, frame_h: int) -> pd.DataFrame:
    """Detection via YOLOv8 (fine-tuned if available, see yolo_detector.py)
    plus a two-stage IoU tracker inspired by ByteTrack (see
    bytetrack_lite.py's docstring for why this replaces Ultralytics' own
    ByteTrack/BoT-SORT integration -- both were buggy in this environment).

    Splits the video into shots first (scene_cut.py) -- real broadcast
    footage cuts between camera angles within seconds, and both
    calibration (one homography per clip) and tracking (identities
    assumed continuous) silently produce nonsense across a cut otherwise.
    Each shot gets its own calibration attempt and a fresh tracker -- but
    team identity (TeamColorAnchor) is deliberately shared across all
    shots, so "team_a" keeps meaning the same real jersey color for the
    whole clip.

    Calibration fallback chain, per shot: (1) the shot's own keypoint
    detection (pitch_calibration_cv.py); (2) if that fails -- expected
    often on short ~100-frame shots, which rarely give the keypoint
    detector enough frames/context to lock on -- reuse the nearest
    *preceding* shot's calibration, since consecutive broadcast shots
    frequently share the same camera framing (e.g. a cut to a close-up
    and back); (3) only if no preceding shot has calibrated yet (e.g. the
    very first shot) fall back to the flat placeholder. Each row records
    which tier produced its calibration (`calib_source`) so this can be
    audited after the fact."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    shots = scene_cut.split_into_shots(video_path)
    print(f"Perception (yolo backend): {total_frames} frames across {len(shots)} shot(s)...")
    team_anchor = team_id.TeamColorAnchor()
    last_valid_calibrator: PitchCalibrator | None = None
    rows = []
    processed = 0
    for shot_i, (start_frame, end_frame) in enumerate(shots):
        print(f"Shot {shot_i + 1}/{len(shots)} (frames {start_frame}-{end_frame})...")
        own_calibrator = _calibrate_shot_own(video_path, start_frame)
        if own_calibrator is not None:
            calibrator, calib_source = own_calibrator, "own"
            last_valid_calibrator = own_calibrator
        elif last_valid_calibrator is not None:
            calibrator, calib_source = last_valid_calibrator, "fallback_prev_shot"
        else:
            calibrator = PitchCalibrator.placeholder_for_frame_size(frame_w, frame_h)
            calib_source = "placeholder"
        shot_rows, processed = _run_yolo_backend_shot(
            video_path, calibrator, fps, start_frame, end_frame, shot_i * _SHOT_TRACK_ID_STRIDE,
            team_anchor, calib_source, processed, total_frames,
        )
        rows.extend(shot_rows)
    print(f"Perception (yolo backend) done: {processed}/{total_frames} frames processed.")
    df = _reclassify_goalkeepers(pd.DataFrame(rows))

    # Two-pass (VAR-style) architecture (2026-07-18, see
    # reports/two_pass_architecture_scoping.md): Pass 1 above is
    # deliberately pose-free -- `find_review_windows` scans its cheap,
    # box-only output (distance/speed + box-aspect-ratio collapse, both
    # already established Layer 3 signals) for short windows worth a
    # closer look, and only THOSE windows get the expensive dual-pass
    # pose estimation Layer 3's pose-dependent triggers need.
    #
    # `_distance_speed_candidates` needs a `speed_mps` column, which is
    # normally a Layer 2 quantity (src/metrics/physical.py) computed AFTER
    # Layer 1 returns -- doesn't exist yet at this point in the pipeline.
    # Borrows that same pure finite-difference calculation for a LOCAL,
    # throwaway copy used only to decide review windows; the real `df`
    # returned to callers is untouched (Layer 2 computes its own
    # `speed_mps` on it the normal way, identically, when `run_metrics`
    # runs next -- this is not a layering shortcut, just reusing one
    # already-cheap, model-free utility function instead of duplicating
    # its math here).
    from src.events.foul_detector.review_windows import find_review_windows
    from src.metrics.physical import add_physical_metrics
    windows = find_review_windows(add_physical_metrics(df))
    print(f"Pass 1 flagged {len(windows)} review window(s) for Pass 2's pose analysis.")
    return _run_pose_pass2(video_path, df, windows, fps)


def run_perception(video_path: str, backend: str = "color") -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if backend == "color":
        from src.perception.synthetic_clip import PX_PER_M
        calibrator = PitchCalibrator.identity_scale(PX_PER_M)
        return _run_color_backend(video_path, calibrator)
    elif backend == "yolo":
        return _run_yolo_backend(video_path, frame_w, frame_h)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


if __name__ == "__main__":
    df = run_perception("data/raw/synthetic_match_clip.mp4", backend="color")
    print(df.head(20))
    print(f"\n{len(df)} rows, {df['track_id'].nunique()} unique tracks")
