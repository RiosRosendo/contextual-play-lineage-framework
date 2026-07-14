"""Camera calibration / pixel-to-pitch homography, per CLAUDE.md section 4
(Layer 1). The real version estimates a homography from detected pitch
keypoints (lines, circle, penalty box corners) against known real-world
pitch coordinates -- not implemented yet (TODO.md). For the skeleton pass
this is a fixed homography:

  - For the synthetic clip, the true pixel-per-meter scale is known exactly
    (it's how the clip was rendered), so calibration is exact.
  - For arbitrary real footage, a flat placeholder scale is used as a
    stand-in and will produce wrong metric units until real calibration
    lands -- this is a known, documented limitation, not a hidden bug.
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
        return float(vec[0]), float(vec[1])
