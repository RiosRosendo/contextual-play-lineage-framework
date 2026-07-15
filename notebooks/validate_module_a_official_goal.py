"""Validates Module A's review-alert logic on real SoccerNet footage using
the match's own official goal timestamp (Labels.json) as the backward-search
trigger, instead of Layer 3's ball-position-crossing heuristic.

Why: goal-line-crossing geometry depends on calibration succeeding during
exactly a goal's own choppiest broadcast seconds (see PROGRESS.md -- this is
precisely where the pipeline's calibration has been failing). That's not the
thing this project is actually trying to validate -- Module A's backward
possession-chain search is. In a real deployment this trigger would come
from an external, reliable source (a goal-line sensor); here, SoccerNet's
own official Labels.json timestamps play that role, and we already have
them (no NDA needed -- Labels.json is freely downloadable).

Match: "2017-01-02 - Sunderland 2 - 2 Liverpool", the same clip used in the
last several PROGRESS.md entries (data/raw/soccernet/foul_before_goal_clip.mp4,
a 100s/2500-frame window, half-clock 37:20-39:00 of the 2nd half).
Labels.json annotations in this window: two "y-card" events (team "away",
i.e. Liverpool) at half-clock 37:49/37:50, and one "soccer-ball" (goal)
event (team "home", i.e. Sunderland) at half-clock 38:43.

Usage:
    python -m notebooks.validate_module_a_official_goal
"""
from __future__ import annotations

from src.assistant.explain import explain_review_alert
from src.events.possession_events import official_goal_event
from src.run_pipeline import run_pipeline

CLIP_PATH = "data/raw/soccernet/foul_before_goal_clip.mp4"
CLIP_START_HALF_CLOCK_S = 37 * 60 + 20  # the trimmed clip starts at half-clock 37:20
OFFICIAL_GOAL_HALF_CLOCK_S = 38 * 60 + 43  # Labels.json: "2 - 38:43", label "soccer-ball", team "home"
OFFICIAL_GOAL_TEAM = "home"
CARD_HALF_CLOCK_S = [37 * 60 + 49, 37 * 60 + 50]  # Labels.json: two "y-card", team "away"


def main() -> None:
    goal_time_in_clip_s = OFFICIAL_GOAL_HALF_CLOCK_S - CLIP_START_HALF_CLOCK_S
    card_times_in_clip_s = [t - CLIP_START_HALF_CLOCK_S for t in CARD_HALF_CLOCK_S]
    goal_event = official_goal_event(goal_time_in_clip_s, team=OFFICIAL_GOAL_TEAM)

    result = run_pipeline(CLIP_PATH, backend="yolo", external_goal_events=[goal_event])
    events = result["events"]
    alerts = result["review_alerts"]
    graph = result["graph"]

    print(f"Official goal trigger: t={goal_time_in_clip_s:.1f}s in-clip (label team: {OFFICIAL_GOAL_TEAM})")
    print(f"Official card(s) (proxy for the foul): t={card_times_in_clip_s} in-clip (label team: away)")
    n_foul = sum(1 for e in events if e["type"] == "foul")
    n_goal = sum(1 for e in events if e["type"] == "goal")
    print(f"\nLayer 3: {len(events)} events total ({n_foul} foul, {n_goal} goal)")
    print(f"Module A: {len(alerts)} review alert(s) raised")

    foul_events = [e for e in events if e["type"] == "foul"]
    near_card = [e for e in foul_events if any(abs(e["time_s"] - t) < 5 for t in card_times_in_clip_s)]
    print(f"\nFoul candidates within 5s of an official card: {len(near_card)}")
    for e in sorted(near_card, key=lambda e: e["time_s"]):
        print(f"  t={e['time_s']:.1f}s  track {e['track_id_a']}({e['team_a']}) vs "
              f"track {e['track_id_b']}({e['team_b']})  closing={e['closing_speed_mps']:.1f} m/s")

    # Always check whether the card-adjacent fouls and the goal share a
    # chain_id -- whether or not an alert fired, this is the real question:
    # a fired alert could still be for the wrong foul (e.g. incidental
    # contact right at the goal itself, in the goal's own short chain,
    # rather than the real foul 53s earlier). Real open play very plausibly
    # contains genuine turnovers in 53s, which would break the chain by
    # Module A's current "same uninterrupted possession chain" definition
    # -- a structural property to surface either way, not just on a miss.
    goal_nodes = [n for n, d in graph.nodes(data=True) if d["type"] == "goal"]
    card_adjacent_chain_ids = {
        d["chain_id"] for n, d in graph.nodes(data=True)
        if d["type"] == "foul" and any(abs(d["time_s"] - t) < 5 for t in card_times_in_clip_s)
    }
    for gn in goal_nodes:
        gdata = graph.nodes[gn]
        print(f"\nGoal node {gn}: chain_id={gdata['chain_id']}, t={gdata['time_s']:.1f}s")
    print(f"Chain_id(s) of the card-adjacent foul candidates: {card_adjacent_chain_ids}")
    turnovers_between = [
        e for e in events
        if e["type"] == "turnover" and min(card_times_in_clip_s) <= e["time_s"] <= goal_time_in_clip_s
    ]
    print(f"Turnovers between the card and the goal: {len(turnovers_between)}")
    for e in turnovers_between:
        print(f"  t={e['time_s']:.1f}s  team={e['team']}")

    for alert in alerts:
        print("\n" + "=" * 70)
        print(explain_review_alert(alert))


if __name__ == "__main__":
    main()
