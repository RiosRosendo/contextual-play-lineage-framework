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
import pandas as pd

from src.perception import (
    color_detector, pitch_calibration_cv, player_classifier, pose_estimator, scene_cut, team_id, yolo_detector,
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
    unreliable position doesn't drive a reclassification decision."""
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
        torso_colors, person_box_tuples = [], []
        for b in boxes:
            if b.cls == "person":
                torso_colors.append(team_id.torso_crop_mean_color(frame, b.x1, b.y1, b.x2, b.y2))
                person_box_tuples.append((b.x1, b.y1, b.x2, b.y2))
        team_labels = team_anchor.assign(torso_colors, person_box_tuples) if torso_colors else []

        det_dicts = []
        person_i = 0
        for b in boxes:
            if b.cls == "person":
                label = team_labels[person_i]
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
                det_cls, team = b.cls, None
            det_dicts.append({
                "cls": det_cls, "team": team, "box": (b.x1, b.y1, b.x2, b.y2), "conf": b.conf,
            })

        tracked = tracker.update(det_dicts)
        for t in tracked:
            cx, cy = (t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2
            x_m, y_m = calibrator.pixel_to_pitch(cx, cy)
            cls = {"person": "player", "referee": "referee", "ball": "ball"}.get(t["cls"], t["cls"])
            if cls == "player":
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
                    cls = "low_confidence"
                elif not (
                    -SIDELINE_MARGIN_M <= x_m <= pitch_calibration_cv.PITCH_LENGTH_M + SIDELINE_MARGIN_M
                    and -SIDELINE_MARGIN_M <= y_m <= pitch_calibration_cv.PITCH_WIDTH_M + SIDELINE_MARGIN_M
                ):
                    # Calibrated position falls well outside the real pitch --
                    # not a player (see SIDELINE_MARGIN_M above). Reclassified
                    # rather than dropped, so this is auditable the same way
                    # calib_source already is.
                    cls = "non_player"
            row = {
                "frame": frame_idx, "time_s": frame_idx / fps, "track_id": t["track_id"] + track_id_offset,
                "cls": cls, "team": t["team"],
                "x": x_m, "y": y_m, "conf": t["conf"], "calib_source": calib_source,
                "box_x1": t["box"][0], "box_y1": t["box"][1], "box_x2": t["box"][2], "box_y2": t["box"][3],
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
