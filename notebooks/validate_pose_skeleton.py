"""Validates the full-body pose-skeleton capability (dual-pass keypoint
extraction in src/perception/pipeline.py + the keypoint signals in
src/events/pose_signals.py) against real footage:

1. Chelsea-Burnley (violent two-player tangle at t~19.6s -- the accepted
   blind spot of every box-geometry signal so far) and Swansea-Man Utd
   (standing tackle at t~16.0s, no fall -- structurally invisible to
   aspect-ratio signals; the earlier per-frame pose exploration saw its
   leg-keypoint distance drop clearly). For each: keypoint coverage, all
   contact-type events, and whether anything fires near the known real
   foul moment.
2. Bonus: handball-candidate scan across every locally downloaded clip.
3. Analytics smoke check: jump and sprint-cadence outputs on both clips.

Usage:
    python -m notebooks.validate_pose_skeleton
"""
from __future__ import annotations

from src.events import pose_signals
from src.metrics.pipeline import run_metrics
from src.perception.pipeline import run_perception

FOUL_JOBS = [
    {"slug": "chelsea_burnley", "path": "data/raw/soccernet/card_chelsea_burnley.mp4",
     "real_foul_time_s": 19.6, "note": "violent two-player tangle"},
    {"slug": "swansea_manutd", "path": "data/raw/soccernet/card_swansea_manutd.mp4",
     "real_foul_time_s": 16.0, "note": "standing tackle, no fall"},
]

HANDBALL_SCAN_CLIPS = [
    "data/raw/soccernet/card_chelsea_burnley.mp4",
    "data/raw/soccernet/card_swansea_manutd.mp4",
    "data/raw/soccernet/card_southampton_liverpool.mp4",
    "data/raw/soccernet/card_mancity_watford.mp4",
    "data/raw/soccernet/card_palace_arsenal.mp4",
    "data/raw/soccernet/clip20s.mp4",
    "data/raw/soccernet/foul_before_goal_clip.mp4",
]

NEAR_WINDOW_S = 3.0

_df_cache: dict = {}


def _player_time_df(path: str):
    if path not in _df_cache:
        perception_df = run_perception(path, backend="yolo")
        _df_cache[path] = run_metrics(perception_df)["player_time_df"]
    return _df_cache[path]


def run_foul_job(job: dict) -> None:
    print(f"\n{'#' * 70}\n{job['slug']} ({job['note']}, real foul t~{job['real_foul_time_s']}s)\n{'#' * 70}")
    df = _player_time_df(job["path"])
    players = df[df["cls"] == "player"]

    n_with_skeleton = players["kp_nose_x"].notna().sum()
    print(f"Keypoint coverage: {n_with_skeleton}/{len(players)} player rows "
          f"({100 * n_with_skeleton / len(players):.0f}%) have a matched skeleton")

    contacts = pose_signals.contact_type_events(df)
    print(f"Contact-type events: {len(contacts)}")
    for c in contacts:
        print(f"  t={c['time_s']:.2f}s  {c['contact_type']:14s}  "
              f"track {c['track_id_a']}({c['team_a']}) -> track {c['track_id_b']}({c['team_b']})  "
              f"dist={c['dist_frac_of_height']:.2f} of height")
    near = [c for c in contacts if abs(c["time_s"] - job["real_foul_time_s"]) <= NEAR_WINDOW_S]
    print(f"Within {NEAR_WINDOW_S:.0f}s of the real foul: {len(near)} "
          f"({sorted(set(c['contact_type'] for c in near))})")

    jumps = pose_signals.jump_events(df)
    sprints = pose_signals.sprint_cadence(df)
    print(f"Analytics: {len(jumps)} jump event(s), {len(sprints)} sprint window(s)")
    for j in jumps[:5]:
        print(f"  jump t={j['time_s']:.2f}s track {j['track_id']} "
              f"rise={j['peak_rise_frac']:.2f} of height (~{j['jump_height_m']:.2f}m)")
    for s in sprints[:5]:
        print(f"  sprint t={s['time_s']:.2f}s track {s['track_id']} "
              f"{s['duration_s']:.1f}s @ {s['mean_speed_mps']:.1f} m/s, {s['steps_per_s']:.1f} steps/s")


def run_handball_scan() -> None:
    print(f"\n{'#' * 70}\nHandball-candidate scan (bonus check)\n{'#' * 70}")
    for path in HANDBALL_SCAN_CLIPS:
        df = _player_time_df(path)
        hits = pose_signals.handball_events(df)
        print(f"{path}: {len(hits)} candidate(s)")
        for h in hits:
            print(f"  t={h['time_s']:.2f}s  track {h['track_id']} ({h['team']})  "
                  f"wrist-ball dist={h['dist_frac_of_height']:.2f} of height")


if __name__ == "__main__":
    for job in FOUL_JOBS:
        run_foul_job(job)
    run_handball_scan()
