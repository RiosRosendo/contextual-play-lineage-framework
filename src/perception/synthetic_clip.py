"""Generates a short synthetic single-camera match clip for end-to-end skeleton
validation while real SoccerNet footage is pending NDA access (see TODO.md /
PROGRESS.md). Renders a top-down pitch with scripted player/ball trajectories
that include one unflagged foul immediately preceding a goal, so Module A's
lineage-graph alert has something real to catch.

This is a stand-in for a broadcast clip, not a broadcast simulator: camera
perspective, occlusion, and realistic player appearance are out of scope here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
PX_PER_M = 8.0
FRAME_W = int(PITCH_LENGTH_M * PX_PER_M)
FRAME_H = int(PITCH_WIDTH_M * PX_PER_M)
FPS = 15
DURATION_S = 10
N_FRAMES = FPS * DURATION_S

TEAM_A_COLOR_BGR = (40, 40, 220)   # red
TEAM_B_COLOR_BGR = (220, 140, 40)  # blue
REF_COLOR_BGR = (30, 220, 220)     # yellow
BALL_COLOR_BGR = (0, 140, 255)     # orange -- distinct from the white pitch lines

PLAYER_RADIUS_PX = 7
BALL_RADIUS_PX = 4


@dataclass
class Actor:
    name: str
    team: str  # "A", "B", "ref", "ball"
    color: tuple
    waypoints: list  # list of (t_s, x_m, y_m)

    def position_at(self, t_s: float) -> tuple[float, float]:
        if t_s <= self.waypoints[0][0]:
            return self.waypoints[0][1], self.waypoints[0][2]
        if t_s >= self.waypoints[-1][0]:
            return self.waypoints[-1][1], self.waypoints[-1][2]
        for (t0, x0, y0), (t1, x1, y1) in zip(self.waypoints, self.waypoints[1:]):
            if t0 <= t_s <= t1:
                frac = (t_s - t0) / (t1 - t0) if t1 > t0 else 0.0
                return x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)
        return self.waypoints[-1][1], self.waypoints[-1][2]


def _script() -> list[Actor]:
    """Scripted scenario: A2 is fouled by B1 near midfield at t=3s (no whistle,
    i.e. unflagged). Play continues in the same possession chain: A1 -> A3, who
    scores at t=8s. Module A should flag this goal for review."""
    return [
        Actor("A1", "A", TEAM_A_COLOR_BGR, [(0, 20, 34), (3, 35, 30), (5, 45, 25), (8, 55, 20), (10, 55, 20)]),
        Actor("A2", "A", TEAM_A_COLOR_BGR, [(0, 30, 40), (3, 42, 34), (3.3, 42.5, 34.2), (5, 40, 36), (10, 38, 36)]),
        Actor("A3", "A", TEAM_A_COLOR_BGR, [(0, 40, 15), (5, 60, 18), (8, 95, 22), (10, 100, 22)]),
        Actor("B1", "B", TEAM_B_COLOR_BGR, [(0, 55, 32), (2.5, 44, 33), (3.3, 42.5, 34.2), (5, 46, 33), (10, 48, 33)]),
        Actor("B2", "B", TEAM_B_COLOR_BGR, [(0, 70, 20), (5, 85, 22), (8, 92, 20), (10, 92, 20)]),
        Actor("B3", "B", TEAM_B_COLOR_BGR, [(0, 90, 34), (10, 90, 34)]),
        Actor("REF", "ref", REF_COLOR_BGR, [(0, 30, 10), (3, 35, 12), (8, 70, 15), (10, 70, 15)]),
        Actor(
            "BALL", "ball", BALL_COLOR_BGR,
            [(0, 30, 40), (3, 42, 34), (5, 45, 25), (6.5, 60, 18), (8, 95, 22), (10, 96, 22)],
        ),
    ]


def _pitch_background() -> np.ndarray:
    img = np.full((FRAME_H, FRAME_W, 3), (60, 140, 60), dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (FRAME_W - 3, FRAME_H - 3), (255, 255, 255), 2)
    cv2.line(img, (FRAME_W // 2, 0), (FRAME_W // 2, FRAME_H), (255, 255, 255), 2)
    cv2.circle(img, (FRAME_W // 2, FRAME_H // 2), int(9.15 * PX_PER_M), (255, 255, 255), 2)
    for gx in (0, FRAME_W):
        cv2.rectangle(
            img,
            (gx - int(9.16 * PX_PER_M) if gx else 0, FRAME_H // 2 - int(20.16 * PX_PER_M)),
            (gx + int(9.16 * PX_PER_M) if not gx else FRAME_W, FRAME_H // 2 + int(20.16 * PX_PER_M)),
            (255, 255, 255), 2,
        )
    return img


def generate_synthetic_clip(out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    actors = _script()
    background = _pitch_background()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (FRAME_W, FRAME_H))

    for frame_idx in range(N_FRAMES):
        t_s = frame_idx / FPS
        frame = background.copy()
        for actor in actors:
            x_m, y_m = actor.position_at(t_s)
            px, py = int(x_m * PX_PER_M), int(y_m * PX_PER_M)
            radius = BALL_RADIUS_PX if actor.team == "ball" else PLAYER_RADIUS_PX
            cv2.circle(frame, (px, py), radius, actor.color, -1)
            if actor.team != "ball":
                cv2.putText(frame, actor.name, (px - 10, py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    return out_path


if __name__ == "__main__":
    path = generate_synthetic_clip("data/raw/synthetic_match_clip.mp4")
    print(f"Wrote synthetic clip to {path} ({N_FRAMES} frames @ {FPS} fps)")
