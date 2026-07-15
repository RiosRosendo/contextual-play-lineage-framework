"""Aggregate report for the foul-verdict-audit capability
(src/assistant/explain.py) across ALL real, isolated single-card incidents
checked so far: the original Sunderland-Liverpool card (from
notebooks/validate_foul_verdict_audit.py) plus the 5 new ones from
notebooks/run_card_clips_audit.py. Produces the small accuracy table
CLAUDE.md's paper needs, explicit about N, instead of the single anecdote.

Also reports a statistical sanity check on the ONE apparent "hit"
(Sunderland-Liverpool): given that clip's own candidate density (31
candidates / 100s), a Poisson back-of-envelope shows a candidate landing
within +/-5s of ANY randomly chosen timestamp in that specific clip was
already ~95% likely by chance alone -- so that hit is weak evidence of the
system genuinely detecting the specific foul, not strong support for the
capability's accuracy. This is deliberately included: reporting the raw hit
count without this caveat would overstate what was actually shown.

Usage:
    python -m notebooks.report_card_clips_audit
"""
from __future__ import annotations

import math

# One row per real single-card instance checked. `n_total_candidates` and
# `n_near_card` are over the full clip and the +/-5s (Sunderland-Liverpool:
# +/-5s too, see validate_foul_verdict_audit.py) window around the card,
# respectively. `verdict_hit`/`severity_hit` are None when there was no
# candidate near the card to even evaluate.
INSTANCES = [
    {
        "match": "Sunderland 2-2 Liverpool", "card": "y-card (away/Liverpool)",
        "clip_len_s": 100, "n_total_candidates": 31, "n_near_card": 3,
        "verdict_hit": True, "severity_hit": True,  # one of the 3 was 'reckless'->yellow, matching exactly
        "note": "1 of 3 near-card candidates matched severity exactly; but see base-rate caveat below.",
    },
    {
        "match": "Chelsea 1-1 Burnley", "card": "y-card (away/Burnley)",
        "clip_len_s": 20, "n_total_candidates": 0, "n_near_card": 0,
        "verdict_hit": False, "severity_hit": False,
        "note": "Real tackle clearly visible on inspection (~t=19.6s, both players on the ground) "
                "but zero contact candidates anywhere in the clip.",
    },
    {
        "match": "Crystal Palace 0-1 Arsenal", "card": "y-card (home/Crystal Palace)",
        "clip_len_s": 20, "n_total_candidates": 8, "n_near_card": 0,
        "verdict_hit": False, "severity_hit": False,
        "note": "All 8 candidates cluster at t=0.0-4.4s (ordinary open play); none within 5.6s of the "
                "card at t=10s -- the candidates found are unrelated to the carded incident.",
    },
    {
        "match": "Swansea 1-1 Manchester United", "card": "y-card (away/Man Utd)",
        "clip_len_s": 20, "n_total_candidates": 0, "n_near_card": 0,
        "verdict_hit": False, "severity_hit": False,
        "note": "Real tackle clearly visible on inspection (~t=16s, standing-pose sliding challenge) "
                "but zero contact candidates anywhere in the clip.",
    },
    {
        "match": "Southampton 0-1 Liverpool", "card": "y-card (home/Southampton)",
        "clip_len_s": 20, "n_total_candidates": 0, "n_near_card": 0,
        "verdict_hit": False, "severity_hit": False,
        "note": "Real contact visible on inspection (~t=4s, player down, teammate/ref converging) "
                "but zero contact candidates anywhere in the clip.",
    },
    {
        "match": "Manchester City 2-0 Watford", "card": "y-card (home/Man City)",
        "clip_len_s": 20, "n_total_candidates": 0, "n_near_card": 0,
        "verdict_hit": False, "severity_hit": False,
        "note": "Zero contact candidates anywhere in the clip; incident likely just outside the "
                "trimmed 20s window or missed by detection during the stoppage.",
    },
]


def poisson_hit_probability(rate_per_s: float, window_s: float) -> float:
    """P(at least 1 event in `window_s`) under a Poisson approximation with
    the clip's own average rate -- i.e. how likely a "hit" near a random
    timestamp would be by chance alone, given how noisy the heuristic
    already is in that specific clip."""
    expected = rate_per_s * window_s
    return 1 - math.exp(-expected)


if __name__ == "__main__":
    n = len(INSTANCES)
    n_verdict_hits = sum(1 for i in INSTANCES if i["verdict_hit"])
    n_severity_hits = sum(1 for i in INSTANCES if i["severity_hit"])

    print(f"Real single-card instances checked: N={n} (1 originally validated + {n - 1} new, across "
          f"{n} different matches)\n")
    print(f"{'Match':<32} {'Card':<24} {'clip_s':>6} {'total':>6} {'near':>5} {'verdict':>8} {'severity':>9}")
    for i in INSTANCES:
        print(f"{i['match']:<32} {i['card']:<24} {i['clip_len_s']:>6} {i['n_total_candidates']:>6} "
              f"{i['n_near_card']:>5} {str(i['verdict_hit']):>8} {str(i['severity_hit']):>9}")

    print(f"\nAggregate: verdict-hit rate = {n_verdict_hits}/{n} ({100 * n_verdict_hits / n:.0f}%); "
          f"severity-hit rate = {n_severity_hits}/{n} ({100 * n_severity_hits / n:.0f}%).")

    print("\nDetail notes:")
    for i in INSTANCES:
        print(f"- {i['match']}: {i['note']}")

    sl = INSTANCES[0]
    rate = sl["n_total_candidates"] / sl["clip_len_s"]
    p_hit_by_chance = poisson_hit_probability(rate, window_s=10.0)
    print(f"\nBase-rate caveat on the one hit (Sunderland-Liverpool): that clip's own candidate rate is "
          f"{rate:.3f}/s ({sl['n_total_candidates']} candidates / {sl['clip_len_s']}s). Under a Poisson "
          f"approximation, P(>=1 candidate within a random 10s window in that specific clip) = "
          f"{p_hit_by_chance:.3f} -- i.e. a 'hit' near ANY randomly chosen timestamp in that clip was "
          f"already ~{100 * p_hit_by_chance:.0f}% likely by chance alone, given how permissive the "
          f"heuristic already is there. This does not disprove the one severity match was meaningful, "
          f"but it substantially weakens it as evidence of genuine detection, rather than base-rate luck.")

    print(
        "\nOverall: across this small, cleaner sample (5 new isolated single-card incidents, no "
        "double-card clusters, 4 different clubs/matches never used before), the contact-candidate "
        "heuristic (src/events/foul_detector/contact_candidates.py) produced ZERO candidates near the "
        "real card in every single new instance -- in 3 of 5, visual inspection confirms the real foul "
        "IS present in the footage, so this is a real miss, not an absence of ground truth to find. "
        "The foul-verdict-audit logic itself (src/assistant/explain.py) was never actually exercised "
        "against a genuine real foul in this new batch, since no candidate ever reached it near the "
        "right moment -- so this is evidence about Layer 3's upstream detection, not about Module C's "
        "reasoning, which behaved correctly whenever it WAS given a real candidate (both here and in "
        "the original Sunderland-Liverpool case)."
    )
