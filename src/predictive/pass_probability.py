"""Layer 4 pass probability. the project spec section 4 targets a GNN over game
state; for the first end-to-end pass this is a hand-set logistic formula
over geometric features (distance, angle change, nearest-defender pressure)
-- same "simplified formula first" philosophy as the other layers. Swapping
in a PyTorch Geometric GNN is tracked in the internal task list.
"""
from __future__ import annotations

import math

import numpy as np

# Hand-tuned coefficients for the logistic formula -- not fit to data yet.
_W_DIST = -0.04
_W_PRESSURE = 0.15
_BIAS = 2.0


def _nearest_defender_dist(target_xy: tuple[float, float], defender_positions: list[tuple[float, float]]) -> float:
    if not defender_positions:
        return 20.0  # no pressure info -> assume open
    dists = [math.dist(target_xy, d) for d in defender_positions]
    return min(dists)


def pass_probability(from_xy: tuple[float, float], to_xy: tuple[float, float],
                      defender_positions: list[tuple[float, float]]) -> float:
    dist = math.dist(from_xy, to_xy)
    pressure = _nearest_defender_dist(to_xy, defender_positions)
    logit = _BIAS + _W_DIST * dist + _W_PRESSURE * pressure
    return float(1.0 / (1.0 + np.exp(-logit)))
