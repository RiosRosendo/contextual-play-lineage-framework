"""Layer 4 expected goals (xG). the project spec section 4 targets gradient
boosting (XGBoost/LightGBM) over geometric shot features; for the first
end-to-end pass this is a hand-set logistic formula over distance and angle
to goal -- the same simplified-formula-first approach used elsewhere.
Training an XGBoost model on real shot data is tracked in the internal task list.
"""
from __future__ import annotations

import math

import numpy as np

GOAL_CENTER = (105.0, 34.0)
GOAL_HALF_WIDTH = 3.66

_W_DIST = -0.10
_W_ANGLE = 1.2
_BIAS = 1.0


def _angle_to_goal(shot_xy: tuple[float, float]) -> float:
    """Angle (radians) subtended by the goal mouth from the shot location --
    larger angle means an easier, more central/closer shot."""
    x, y = shot_xy
    post1 = (GOAL_CENTER[0], GOAL_CENTER[1] - GOAL_HALF_WIDTH)
    post2 = (GOAL_CENTER[0], GOAL_CENTER[1] + GOAL_HALF_WIDTH)
    v1 = (post1[0] - x, post1[1] - y)
    v2 = (post2[0] - x, post2[1] - y)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag = math.hypot(*v1) * math.hypot(*v2)
    if mag == 0:
        return 0.0
    return math.acos(max(-1.0, min(1.0, dot / mag)))


def expected_goal_probability(shot_xy: tuple[float, float]) -> float:
    dist = math.dist(shot_xy, GOAL_CENTER)
    angle = _angle_to_goal(shot_xy)
    logit = _BIAS + _W_DIST * dist + _W_ANGLE * angle
    return float(1.0 / (1.0 + np.exp(-logit)))
