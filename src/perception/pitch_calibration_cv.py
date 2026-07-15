"""Classical-CV pitch calibration: detects real pitch keypoints in a frame
and computes a homography to real-world pitch coordinates, replacing the
fixed-guess placeholders in calibration.py -- CLAUDE.md section 4.

Two strategies, tried in order by `calibrate_frame`:

1. `calibrate_goal_area`: for a broadcast shot showing one goal (the common
   "tactical" camera framing, e.g. SoccerSum). Detects the goalposts'
   ground-contact points (a very distinctive, unambiguous shape: two tall
   bright near-vertical bars joined by a crossbar) as two known points
   (depth=0, lateral=+-3.66m from goal center), then finds the strongest
   pitch line in the goal area and evaluates it at the two posts' x
   positions to get two more points at a second, known depth. Assumes that
   line is the penalty box's back line (16.5m) -- a named, documented
   simplification: broadcast shots most commonly frame the penalty box
   prominently, but this is a guess, not a verified classification, and
   will be wrong if a different line (e.g. the 6-yard box) was detected
   instead.

   Important, measured limitation: all 4 points this strategy fits from
   are clustered in a small region near the goal, so the homography is
   only locally well-conditioned. Checked directly: transforming a point
   on the near touchline (roughly 250-350px further from the goal than
   the fitted region) gave wildly inconsistent, sometimes sign-flipped
   lateral coordinates depending which pixel on that same straight line
   was queried -- i.e. it's not reliable far from the goal area it was
   calibrated against. Metrics computed near the goal should be reasonable;
   metrics for play happening elsewhere on the pitch will not be.

2. `calibrate_full_pitch_boundary`: for a shot where the whole pitch is
   visible in one frame (e.g. SoccerTrack v2's full-pitch panorama).
   Approximates the pitch (green-mask) contour as a quadrilateral and maps
   its 4 corners to the real pitch's 4 corners. Documented limitation:
   SoccerTrack v2's panorama has visible fisheye/stitching distortion, so
   the pitch boundary isn't a true quadrilateral in image space -- this
   strategy still fits a homography, but it will not fully correct that
   distortion (a pinhole/homography model can't represent fisheye lens
   distortion), so accuracy degrades away from the frame center. Better
   than the flat placeholder guess, not a substitute for real undistortion.

Both return `None` if their required features aren't confidently detected,
so callers can fall back to the placeholder -- matching the existing
fallback pattern in yolo_detector.py.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.perception.calibration import PitchCalibrator

GOAL_HALF_WIDTH_M = 3.66  # goal width 7.32m
ASSUMED_BOX_DEPTH_M = 16.5  # penalty box -- see module docstring caveat

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


def _pitch_mask(frame_bgr: np.ndarray) -> np.ndarray | None:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 60, 40), (90, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.1 * frame_bgr.shape[0] * frame_bgr.shape[1]:
        return None
    mask = np.zeros(frame_bgr.shape[:2], np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return cv2.erode(mask, np.ones((7, 7), np.uint8)), largest


def _line_mask(frame_bgr: np.ndarray, exclude_mask: np.ndarray, pitch_mask: np.ndarray) -> np.ndarray:
    """Top-hat filter: highlights thin bright pitch lines by *local*
    contrast against the grass, which works even when the lines are faint/
    worn (a flat brightness threshold picks up player jerseys instead --
    see PROGRESS.md for how this was found)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(tophat, 20, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, mask, mask=exclude_mask)
    mask = cv2.bitwise_and(mask, mask, mask=pitch_mask)
    return mask


def _person_exclusion_mask(frame_shape: tuple, person_boxes: list) -> np.ndarray:
    h, w = frame_shape[:2]
    mask = np.ones((h, w), np.uint8) * 255
    pad = 6
    for x1, y1, x2, y2 in person_boxes:
        cv2.rectangle(
            mask, (max(0, int(x1) - pad), max(0, int(y1) - pad)),
            (min(w, int(x2) + pad), min(h, int(y2) + pad)), 0, -1,
        )
    return mask


def _detect_goal(frame_bgr: np.ndarray) -> tuple | None:
    """Finds the goal frame (posts + crossbar) as one bright connected
    blob, then locates the two post columns as the two local maxima of
    "how far down this column stays white" within the blob. Returns
    ((left_x, left_bottom_y), (right_x, right_bottom_y)) or None."""
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 190), (180, 50, 255))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # goal shape: a wide, moderately tall, sparsely-filled (post+crossbar,
        # not a solid rectangle) blob.
        if cw > 0.03 * w and ch > 0.02 * h and area > 200 and 0.05 < area / (cw * ch) < 0.7:
            candidates.append((area, x, y, cw, ch, c))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, x, y, cw, ch, c = candidates[0]

    blob = np.zeros((h, w), np.uint8)
    cv2.drawContours(blob, [c], -1, 255, -1)
    col_bottoms = []
    for col in range(x, x + cw):
        ys = np.where(blob[y:y + ch, col] > 0)[0]
        if len(ys):
            col_bottoms.append((col, y + int(ys.max())))
    if len(col_bottoms) < 2:
        return None

    # left post: deepest column in the left half of the blob; right post:
    # deepest column in the right half. Avoids picking two columns from the
    # same post if one post is much more prominent than the other.
    mid = x + cw / 2
    left_candidates = [c for c in col_bottoms if c[0] < mid]
    right_candidates = [c for c in col_bottoms if c[0] >= mid]
    if not left_candidates or not right_candidates:
        return None
    left = max(left_candidates, key=lambda c: c[1])
    right = max(right_candidates, key=lambda c: c[1])
    return left, right


def _strongest_line_near(line_mask: np.ndarray, x_range: tuple, y_range: tuple) -> tuple | None:
    """Returns (slope, intercept) of the longest Hough line segment whose
    midpoint falls within the given region, fit as y = slope*x + intercept."""
    lines = cv2.HoughLinesP(line_mask, 1, np.pi / 180, threshold=40, minLineLength=60, maxLineGap=15)
    if lines is None:
        return None
    best, best_len = None, 0
    for l in lines:
        x1, y1, x2, y2 = l.ravel().tolist()
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if not (x_range[0] <= mx <= x_range[1] and y_range[0] <= my <= y_range[1]):
            continue
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if x2 == x1:
            continue
        if length > best_len:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            best, best_len = (slope, intercept), length
    return best


def calibrate_goal_area(frame_bgr: np.ndarray, person_boxes: list) -> PitchCalibrator | None:
    goal = _detect_goal(frame_bgr)
    if goal is None:
        return None
    (lx, ly), (rx, ry) = goal

    pitch = _pitch_mask(frame_bgr)
    if pitch is None:
        return None
    pitch_mask, _ = pitch
    exclude = _person_exclusion_mask(frame_bgr.shape, person_boxes)
    lines = _line_mask(frame_bgr, exclude, pitch_mask)

    h, w = frame_bgr.shape[:2]
    # Search only a bounded region below the goal, not all the way to the
    # bottom of the frame -- too wide a window here previously let a much
    # longer, unrelated line (the near touchline) outcompete the actual
    # box line just for being longer. The box back line is expected within
    # roughly one goal-height's distance below the goal mouth.
    goal_height_px = abs(ly - ry) + 60
    search_depth_px = max(150, 3 * goal_height_px)
    goal_line = _strongest_line_near(
        lines, x_range=(0, w), y_range=(min(ly, ry), min(ly, ry) + search_depth_px),
    )
    if goal_line is None:
        return None
    slope, intercept = goal_line
    line_left_y = slope * lx + intercept
    line_right_y = slope * rx + intercept

    image_pts = np.array([[lx, ly], [rx, ry], [lx, line_left_y], [rx, line_right_y]], dtype=np.float32)
    world_pts = np.array([
        [0.0, -GOAL_HALF_WIDTH_M], [0.0, GOAL_HALF_WIDTH_M],
        [ASSUMED_BOX_DEPTH_M, -GOAL_HALF_WIDTH_M], [ASSUMED_BOX_DEPTH_M, GOAL_HALF_WIDTH_M],
    ], dtype=np.float32)
    h_matrix, _ = cv2.findHomography(image_pts, world_pts)
    if h_matrix is None:
        return None
    return PitchCalibrator(h_matrix)


def calibrate_full_pitch_boundary(frame_bgr: np.ndarray) -> PitchCalibrator | None:
    pitch = _pitch_mask(frame_bgr)
    if pitch is None:
        return None
    _, contour = pitch
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) != 4:
        return None
    pts = approx.reshape(-1, 2).astype(np.float32)
    # order by angle around the centroid so corners are in a consistent
    # (clockwise) order regardless of contour winding
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]

    world_pts = np.array([
        [0.0, 0.0], [PITCH_LENGTH_M, 0.0],
        [PITCH_LENGTH_M, PITCH_WIDTH_M], [0.0, PITCH_WIDTH_M],
    ], dtype=np.float32)
    h_matrix, _ = cv2.findHomography(pts, world_pts)
    if h_matrix is None:
        return None
    return PitchCalibrator(h_matrix)


def _looks_like_wide_shot(frame_bgr: np.ndarray) -> bool:
    """If the pitch mask's bounding box covers most of the frame, this is a
    full-pitch/panorama-style view, not a tight goal-area shot -- prefer
    calibrate_full_pitch_boundary in that case. Added after finding that
    calibrate_goal_area produced a false-positive "goal" match on a
    full-pitch panorama frame (some other bright structure coincidentally
    fit the shape heuristic), which would otherwise have silently won since
    it's tried first."""
    pitch = _pitch_mask(frame_bgr)
    if pitch is None:
        return False
    mask, _ = pitch
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return False
    frame_h, frame_w = frame_bgr.shape[:2]
    width_frac = (xs.max() - xs.min()) / frame_w
    height_frac = (ys.max() - ys.min()) / frame_h
    return width_frac > 0.85 and height_frac > 0.5


def calibrate_frame(frame_bgr: np.ndarray, person_boxes: list) -> PitchCalibrator | None:
    if _looks_like_wide_shot(frame_bgr):
        calib = calibrate_full_pitch_boundary(frame_bgr)
        if calib is not None:
            return calib
        return calibrate_goal_area(frame_bgr, person_boxes)

    calib = calibrate_goal_area(frame_bgr, person_boxes)
    if calib is not None:
        return calib
    return calibrate_full_pitch_boundary(frame_bgr)
