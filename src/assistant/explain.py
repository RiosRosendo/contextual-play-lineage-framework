"""Module C entry point: given a Module A review alert, retrieves the
relevant IFAB excerpt and generates a grounded natural-language explanation
-- the project spec section 4 ("generates a grounded natural-language explanation
without hallucinating the citation").

No external LLM call is made by default: the explanation is template-filled
directly from the retrieved excerpt and event metadata, so every sentence
traces back to a concrete source (the event dict or the local corpus). This
also means the skeleton runs with no API key configured. Wiring an actual
LLM call to rephrase this (kept strictly grounded to the same retrieved
excerpt) is a the internal task list item, off by default.

Also provides a second, independent capability: `assess_foul_candidate` /
`explain_foul_candidate` audit a single Layer 3 foul candidate on its own
terms (foul/no-foul, and a severity/card recommendation), grounded the same
way. This is deliberately standalone -- no goal, no possession chain, no
Module A graph traversal -- so it can be validated against a referee's
actual decision (e.g. SoccerNet's official card labels) for any contact
event, not only ones that happen to precede a goal.
"""
from __future__ import annotations

from src.assistant.retriever import get_retriever
from src.events.foul_detector.contact_candidates import MAX_CLOSING_SPEED_MPS

_QUERY_BY_CONTEXT = "foul careless reckless excessive force direct free kick"

# Closing speed is the only contact-intensity signal Layer 3 already
# computes, so it stands in here for the biomechanics-based severity
# judgment a real video encoder would make (the project spec section 3: simplest
# version first, not a real severity classifier). Thresholds are a
# deliberately round, illustrative split of contact_candidates.py's
# existing plausible range (3-12 m/s combined closing speed).
CARELESS_MAX_CLOSING_SPEED_MPS = 6.0
RECKLESS_MAX_CLOSING_SPEED_MPS = 9.0  # above this: excessive force

_SEVERITY_QUERY = {
    "careless": "careless lack of attention no further sanction",
    "reckless": "reckless disregard danger opponent caution yellow card",
    "excessive_force": "excessive force far more than necessary sent off red card",
}
_SEVERITY_CARD = {"careless": "none", "reckless": "yellow", "excessive_force": "red"}
_SEVERITY_ORDER = ("careless", "reckless", "excessive_force")

# Human-readable label and offense-category retrieval query per keypoint
# contact type (src/events/pose_signals.py's CONTACT_RULES), so the verdict
# can name and ground *which* Law 12 offense the contact resembles, not
# just how fast the players were closing.
_CONTACT_TYPE_LABEL = {
    "hand_to_face": "a hand/arm strike to the head or face",
    "elbow_to_body": "an elbow strike to the body",
    "shirt_pull": "holding (sustained shirt-pull)",
    "leg_contact": "a leg/foot challenge",
}
_CONTACT_TYPE_QUERY = {
    "hand_to_face": "striking or attempting to strike",
    "elbow_to_body": "striking or attempting to strike",
    "shirt_pull": "holding an opponent",
    "leg_contact": "tackling or challenging tripping kicking",
}

# Contact landing on the head/face is treated by Law 12 as inherently
# dangerous regardless of how fast the players were closing (a slow arm can
# still injure) -- "excessive force"/"reckless" is about danger and force,
# not velocity alone. These two contact types set a severity FLOOR: the
# closing-speed proxy can still escalate a hand/elbow strike above this
# floor (a very high closing speed still means excessive force), but never
# reduce it below "reckless" purely because the players were moving slowly.
_SEVERITY_FLOOR_BY_CONTACT_TYPE = {
    "hand_to_face": "reckless",
    "elbow_to_body": "reckless",
}

# dist_frac_of_height (src/events/pose_signals.py: closest joint-to-joint
# distance, as a fraction of the players' mean box height) is only ever
# populated by the keypoint-contact trigger, and only ever within its own
# trigger range [0, CONTACT_FRAC_OF_HEIGHT=0.2] -- a smaller value means the
# contact point was closer/more direct relative to the players' own size.
# Used ONLY as a severity fallback when closing_speed_mps is implausible
# (see assess_foul_candidate) and no contact-type floor applies; a purely
# geometric signal, not subject to the tracking-corruption failure mode that
# makes closing speed untrustworthy during violent contact. Thresholds are a
# deliberately round split of that [0, 0.2] range, the same illustrative
# spirit as CARELESS/RECKLESS_MAX_CLOSING_SPEED_MPS above.
DIST_FRAC_EXCESSIVE_MAX = 0.07
DIST_FRAC_RECKLESS_MAX = 0.14


def _severity_for(closing_speed_mps: float) -> str:
    if closing_speed_mps <= CARELESS_MAX_CLOSING_SPEED_MPS:
        return "careless"
    if closing_speed_mps <= RECKLESS_MAX_CLOSING_SPEED_MPS:
        return "reckless"
    return "excessive_force"


def _severity_from_dist_frac(dist_frac_of_height: float) -> str:
    if dist_frac_of_height <= DIST_FRAC_EXCESSIVE_MAX:
        return "excessive_force"
    if dist_frac_of_height <= DIST_FRAC_RECKLESS_MAX:
        return "reckless"
    return "careless"


def _contact_type_floor(contact_types: list[str]) -> str | None:
    """The highest severity floor implied by any identified contact type, or
    None if none of them carry one (see _SEVERITY_FLOOR_BY_CONTACT_TYPE)."""
    floor = None
    for ctype in contact_types:
        candidate = _SEVERITY_FLOOR_BY_CONTACT_TYPE.get(ctype)
        if candidate and (floor is None or _SEVERITY_ORDER.index(candidate) > _SEVERITY_ORDER.index(floor)):
            floor = candidate
    return floor


def _apply_contact_type_floor(severity: str, contact_types: list[str]) -> tuple[str, bool]:
    """Raises `severity` to the highest floor implied by any identified
    contact type, if that floor exceeds the closing-speed-derived severity.
    Returns (severity, was_raised)."""
    floor = _contact_type_floor(contact_types)
    if floor is not None and _SEVERITY_ORDER.index(floor) > _SEVERITY_ORDER.index(severity):
        return floor, True
    return severity, False


def assess_foul_candidate(event: dict) -> dict:
    """Independently classifies one Layer 3 foul-candidate event as
    foul/no-foul (and a severity + recommended card if a foul), grounded in
    a retrieved IFAB Law 12 excerpt. Returns structured data (not just
    text) so a caller can compare the verdict against ground truth (e.g. a
    real referee decision) programmatically.

    Every event reaching this function already passed Layer 3's physical
    contact-candidate heuristic (contact_candidates.py), so -- consistent
    with how Module A's own lineage graph already treats these events (see
    lineage/graph.py's docstring) -- this verdict does NOT gate on the
    classifier's `is_foul`/`foul_probability`: that head is untrained, and
    real-footage validation confirmed its probability isn't currently
    discriminating anything meaningful here (every one of 31 candidates on
    a real 100s clip scored 0.18-0.48, i.e. below the nominal 0.5
    threshold with no exceptions -- a property of random initialization,
    not evidence none of the contacts were fouls). Once the classifier is
    trained (the internal task list), gating the verdict on `is_foul` becomes meaningful
    and should replace this.

    If the event carries a `contact_types` annotation (from
    src/events/pose_signals.annotate_foul_contact_types), it is folded into
    the severity judgment alongside closing speed: a hand-to-face or
    elbow-to-body contact sets a "reckless" floor regardless of speed (see
    _SEVERITY_FLOOR_BY_CONTACT_TYPE), and the specific offense category
    (striking / holding / tackling) is retrieved separately so the
    explanation can name which Law 12 offense the contact resembles, not
    just its severity tier. Events with no contact-type annotation (or an
    empty one) fall back to closing-speed-only reasoning, unchanged.

    `closing_speed_mps` is NOT trusted for severity above
    MAX_CLOSING_SPEED_MPS (contact_candidates.py's own physical-plausibility
    cap): the keypoint-contact trigger deliberately lets candidates through
    at implausible speeds (real contact corrupts the box-derived speed
    estimate, the same failure already on record for pose-collapse/
    torso-fall), but that same corrupted number must not then drive the
    severity verdict. Above the cap, severity instead falls back to a
    hierarchy: the contact-type floor if one applies, else
    `dist_frac_of_height` (a purely geometric signal, unaffected by speed
    corruption) if the event carries one, else a conservative "careless"
    default (disclosed via `severity_source`, not silently guessed)."""
    retriever = get_retriever()
    contact_types = event.get("contact_types") or []
    closing_speed = event["closing_speed_mps"]
    speed_plausible = closing_speed <= MAX_CLOSING_SPEED_MPS

    if speed_plausible:
        severity = _severity_for(closing_speed)
        severity, raised_by_contact_type = _apply_contact_type_floor(severity, contact_types)
        severity_source = "contact_type_floor" if raised_by_contact_type else "closing_speed"
    else:
        floor = _contact_type_floor(contact_types)
        dist_frac = event.get("dist_frac_of_height")
        if floor is not None:
            severity, severity_source = floor, "contact_type_floor"
        elif dist_frac is not None:
            severity, severity_source = _severity_from_dist_frac(dist_frac), "dist_frac_of_height"
        else:
            severity, severity_source = "careless", "unresolved_implausible_speed"

    excerpt = retriever.retrieve(_SEVERITY_QUERY[severity], k=1)[0]

    contact_excerpt = None
    for ctype in contact_types:
        query = _CONTACT_TYPE_QUERY.get(ctype)
        if query is not None:
            contact_excerpt = retriever.retrieve(query, k=1)[0]
            break

    return {"verdict": "foul", "severity": severity, "recommended_card": _SEVERITY_CARD[severity],
            "contact_types": contact_types, "severity_source": severity_source,
            "closing_speed_plausible": speed_plausible,
            "excerpt": excerpt, "contact_excerpt": contact_excerpt, "event": event}


def explain_foul_candidate(event: dict) -> str:
    """Formats `assess_foul_candidate`'s verdict as grounded natural
    language -- same "no hallucinated citation" contract as
    `explain_review_alert`."""
    assessment = assess_foul_candidate(event)
    excerpt = assessment["excerpt"]
    speed_note = "" if assessment["closing_speed_plausible"] else (
        f" (exceeds the {MAX_CLOSING_SPEED_MPS:.0f} m/s plausibility cap -- "
        f"not used for severity, see below)"
    )
    header = (
        f"Contact at t={event['time_s']:.2f}s between track #{event['track_id_a']} "
        f"({event['team_a']}) and track #{event['track_id_b']} ({event['team_b']}), "
        f"closing speed {event['closing_speed_mps']:.1f} m/s{speed_note} "
        f"(illustrative model confidence: {event['foul_probability']:.2f}, classifier untrained)."
    )
    if assessment["verdict"] == "no_foul":
        verdict_line = "VERDICT: no foul -- contact did not meet the foul criteria below."
    else:
        card_label = {"none": "no card", "yellow": "yellow card (caution)",
                      "red": "red card (sent off)"}[assessment["recommended_card"]]
        verdict_line = f"VERDICT: foul, severity '{assessment['severity']}' -> recommended sanction: {card_label}."

    contact_types = assessment["contact_types"]
    if contact_types:
        labels = ", ".join(_CONTACT_TYPE_LABEL.get(c, c) for c in contact_types)
        contact_line = f"Contact type(s) identified at the keypoint level: {labels}."
    else:
        contact_line = "Contact type(s) identified at the keypoint level: none resolved."

    severity_source = assessment["severity_source"]
    if severity_source == "contact_type_floor":
        contact_line += (
            " Severity set by contact type: contact to the head/face is treated as inherently "
            "dangerous regardless of closing speed, so it cannot be judged 'careless' on speed alone."
        )
    elif severity_source == "dist_frac_of_height":
        contact_line += (
            " Closing speed was not usable here (see above), so severity falls back to how close "
            "the contact point was, relative to the players' own size -- a purely geometric signal, "
            "not a biomechanics-based force estimate."
        )
    elif severity_source == "unresolved_implausible_speed":
        contact_line += (
            " Closing speed was not usable here and no other severity signal was available, so "
            "severity defaults to the most conservative tier ('careless') rather than guessing."
        )
    else:
        contact_line += " Severity is based on closing speed alone."

    grounding = f"Grounding ({excerpt['law']} -- {excerpt['title']}):\n  \"{excerpt['text']}\""
    contact_excerpt = assessment["contact_excerpt"]
    if contact_excerpt is not None and contact_excerpt["id"] != excerpt["id"]:
        grounding += (
            f"\n\nOffense-category grounding ({contact_excerpt['law']} -- {contact_excerpt['title']}):\n"
            f"  \"{contact_excerpt['text']}\""
        )

    return (
        f"{verdict_line}\n{header}\n{contact_line}\n\n"
        f"{grounding}\n\n"
        f"Note: severity combines a simplified closing-speed proxy (the project spec section 3) "
        f"with the identified contact type where available -- neither is a biomechanics-based "
        f"judgment, and the excerpts above are paraphrased summaries, not verbatim official IFAB "
        f"text -- see src/assistant/ifab_corpus.py."
    )


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
