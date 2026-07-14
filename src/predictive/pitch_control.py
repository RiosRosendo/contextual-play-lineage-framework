"""Layer 4 pitch control. CLAUDE.md section 4 targets a proper spatial
dominance surface (Spearman et al.-style influence functions); the skeleton
version is a coarse time-to-reach grid: each cell is "controlled" by
whichever team could get a player there first, assuming a flat max speed.
Simplified-formula-first, per section 3.
"""
from __future__ import annotations

import numpy as np

ASSUMED_MAX_SPEED_MPS = 8.0


def pitch_control_grid(player_positions: list[tuple[str, float, float]],
                        pitch_length_m: float = 105.0, pitch_width_m: float = 68.0,
                        bins: tuple[int, int] = (21, 14)) -> np.ndarray:
    """player_positions: list of (team, x, y). Returns a (bins[0], bins[1])
    array in [-1, 1]: positive favors team_a, negative favors team_b, scaled
    by the time-to-reach gap between the two teams' nearest player."""
    xs = np.linspace(0, pitch_length_m, bins[0])
    ys = np.linspace(0, pitch_width_m, bins[1])
    grid = np.zeros(bins)

    team_a = [(x, y) for team, x, y in player_positions if team == "team_a"]
    team_b = [(x, y) for team, x, y in player_positions if team == "team_b"]

    for i, gx in enumerate(xs):
        for j, gy in enumerate(ys):
            t_a = _min_time_to_reach(team_a, gx, gy)
            t_b = _min_time_to_reach(team_b, gx, gy)
            gap = t_b - t_a  # positive -> team_a reaches first
            grid[i, j] = np.tanh(gap / 2.0)  # squash to [-1, 1]
    return grid


def _min_time_to_reach(positions: list[tuple[float, float]], gx: float, gy: float) -> float:
    if not positions:
        return float("inf")
    dists = [((x - gx) ** 2 + (y - gy) ** 2) ** 0.5 for x, y in positions]
    return min(dists) / ASSUMED_MAX_SPEED_MPS
