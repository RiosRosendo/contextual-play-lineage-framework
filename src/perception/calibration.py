"""Camera calibration / pixel-to-pitch homography, per CLAUDE.md section 4
(Layer 1).

  - For the synthetic clip, the true pixel-per-meter scale is known exactly
    (it's how the clip was rendered), so calibration is exact.
  - For real footage, `src/perception/pitch_calibration_cv.py` detects
    actual pitch keypoints (goalposts, box lines, or the full pitch
    boundary) and fits a real homography via `PitchCalibrator(h_matrix)`.
  - `placeholder_for_frame_size` (a flat, unvalidated 60x40m guess) is now
    only a last-resort fallback when keypoint detection fails on a given
    frame -- it will produce wrong metric units, a known, documented
    limitation, not a hidden bug.
"""
from __future__ import annotations

import numpy as np


class PitchCalibrator:
    def __init__(self, homography: np.ndarray):
        self.h = homography

    @classmethod
    def identity_scale(cls, px_per_m: float) -> "PitchCalibrator":
        h = np.array([
            [1.0 / px_per_m, 0.0, 0.0],
            [0.0, 1.0 / px_per_m, 0.0],
            [0.0, 0.0, 1.0],
        ])
        return cls(h)

    @classmethod
    def placeholder_for_frame_size(cls, frame_w: int, frame_h: int) -> "PitchCalibrator":
        """Naive placeholder for real footage: assumes the visible frame
        spans a fixed 60m x 40m patch of pitch. Real calibration (keypoint
        homography) replaces this -- see module docstring."""
        px_per_m_x = frame_w / 60.0
        px_per_m_y = frame_h / 40.0
        h = np.array([
            [1.0 / px_per_m_x, 0.0, 0.0],
            [0.0, 1.0 / px_per_m_y, 0.0],
            [0.0, 0.0, 1.0],
        ])
        return cls(h)

    def pixel_to_pitch(self, px: float, py: float) -> tuple[float, float]:
        vec = self.h @ np.array([px, py, 1.0])
        vec = vec / vec[2]  # perspective division; a no-op for the affine-only
        return float(vec[0]), float(vec[1])  # matrices above, whose bottom row is [0,0,1]
