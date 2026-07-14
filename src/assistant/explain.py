"""Module C entry point: given a Module A review alert, retrieves the
relevant IFAB excerpt and generates a grounded natural-language explanation
-- CLAUDE.md section 4 ("generates a grounded natural-language explanation
without hallucinating the citation").

No external LLM call is made by default: the explanation is template-filled
directly from the retrieved excerpt and event metadata, so every sentence
traces back to a concrete source (the event dict or the local corpus). This
also means the skeleton runs with no API key configured. Wiring an actual
LLM call to rephrase this (kept strictly grounded to the same retrieved
excerpt) is a TODO.md item, off by default.
"""
from __future__ import annotations

from src.assistant.retriever import get_retriever

_QUERY_BY_CONTEXT = "foul careless reckless excessive force direct free kick"


def explain_review_alert(alert: dict) -> str:
    goal = alert["goal_event"]
    fouls = alert["foul_events"]
    retriever = get_retriever()
    top_excerpt = retriever.retrieve(_QUERY_BY_CONTEXT, k=1)[0]

    foul_lines = []
    for f in fouls:
        foul_lines.append(
            f"  - Unflagged contact at t={f['time_s']:.2f}s between track "
            f"#{f['track_id_a']} ({f['team_a']}) and track #{f['track_id_b']} ({f['team_b']}), "
            f"closing speed {f['closing_speed_mps']:.1f} m/s "
            f"(illustrative model confidence: {f['foul_probability']:.2f}, classifier untrained)."
        )

    return (
        f"REVIEW ALERT: goal by {goal['team']} at t={goal['time_s']:.2f}s follows an "
        f"unflagged foul in the same possession sequence.\n\n"
        f"Foul(s) found in the possession chain leading to this goal:\n"
        + "\n".join(foul_lines) + "\n\n"
        f"Grounding ({top_excerpt['law']} -- {top_excerpt['title']}):\n"
        f"  \"{top_excerpt['text']}\"\n\n"
        f"Note: the excerpt above is a paraphrased summary for this skeleton, not the "
        f"verbatim official IFAB text -- see src/assistant/ifab_corpus.py."
    )
