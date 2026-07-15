"""Validates Module C's independent foul-verdict/referee-decision-audit
capability (`assess_foul_candidate` / `explain_foul_candidate` in
src/assistant/explain.py) against real referee decisions.

This is deliberately separate from Module A's goal-consequence linking
(build_lineage_graph / find_review_alerts): it audits ANY Layer 3 foul
candidate on its own terms -- foul/no-foul, and a severity/card
recommendation, grounded in IFAB Law 12 -- with no goal and no possession
chain involved at all.

Ground truth: SoccerNet's official Labels.json for this match records two
"y-card" (yellow card) events at half-clock 37:49/37:50, team "away"
(Liverpool) -- see notebooks/validate_module_a_official_goal.py for the
same match/clip's goal-side validation. In-clip (the clip starts at
half-clock 37:20) these are t=29s and t=30s.

Usage:
    python -m notebooks.validate_foul_verdict_audit
"""
from __future__ import annotations

from src.assistant.explain import assess_foul_candidate, explain_foul_candidate
from src.events.pipeline import run_events
from src.metrics.pipeline import run_metrics
from src.perception.pipeline import run_perception

CLIP_PATH = "data/raw/soccernet/foul_before_goal_clip.mp4"
CARD_TIMES_IN_CLIP_S = [29.0, 30.0]  # Labels.json: two "y-card", team "away", half-clock 37:49/37:50
CARD_WINDOW_S = 5.0  # how close a foul candidate must be to a card to count as "near" it


def main() -> None:
    perception_df = run_perception(CLIP_PATH, backend="yolo")
    metrics = run_metrics(perception_df)
    events = run_events(metrics)

    foul_events = [e for e in events if e["type"] == "foul"]
    print(f"Layer 3 foul candidates in the clip: {len(foul_events)}")

    near_card = [
        e for e in foul_events
        if any(abs(e["time_s"] - t) <= CARD_WINDOW_S for t in CARD_TIMES_IN_CLIP_S)
    ]
    print(f"Foul candidates within {CARD_WINDOW_S:.0f}s of an official card: {len(near_card)}\n")

    assessments = []
    for e in sorted(near_card, key=lambda e: e["time_s"]):
        assessment = assess_foul_candidate(e)
        assessments.append(assessment)
        print("=" * 70)
        print(explain_foul_candidate(e))
        print()

    any_foul_verdict = any(a["verdict"] == "foul" for a in assessments)
    any_yellow_match = any(a["recommended_card"] == "yellow" for a in assessments)

    print("=" * 70)
    print("SUMMARY")
    print("Official ground truth: a yellow card WAS given (team 'away') at t~29-30s.")
    print(f"System verdict near that window: "
          f"{'at least one FOUL verdict' if any_foul_verdict else 'NO foul verdicts'} "
          f"among {len(near_card)} candidate(s) checked -> "
          f"{'AGREES' if any_foul_verdict else 'DISAGREES'} directionally with the card.")
    if any_yellow_match:
        print("Stronger match: at least one candidate's own severity verdict is 'reckless' "
              "-> recommended yellow card, matching the real sanction exactly (not just "
              "foul/no-foul agreement).")
    print(
        "Caveat: this checks timestamp proximity only. The system's `team_a`/`team_b` "
        "labels are assigned arbitrarily per clip by TeamColorAnchor and are not mapped "
        "to SoccerNet's 'home'/'away' encoding, so which side (not just whether a foul "
        "occurred) cannot currently be cross-checked against the card's own team label -- "
        "flagging this rather than assuming a match."
    )

    # Also report the full severity/card distribution over ALL foul candidates
    # in the clip (not just the ones near this one known card), since this is
    # useful context on how the severity heuristic behaves in general.
    print(f"\nSeverity/card distribution over all {len(foul_events)} foul candidates in the clip:")
    counts: dict[str, int] = {}
    for e in foul_events:
        a = assess_foul_candidate(e)
        key = a["verdict"] if a["verdict"] == "no_foul" else f"foul/{a['severity']}"
        counts[key] = counts.get(key, 0) + 1
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
