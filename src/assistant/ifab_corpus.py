"""Small local excerpt corpus for Module C's RAG retrieval.

IMPORTANT: these are paraphrased summaries written for this skeleton, NOT
verbatim quotes from the official IFAB Laws of the Game. Presenting them as
official text would risk a hallucinated citation. Before any real use, this
corpus must be replaced with actual extracted text from the official IFAB
PDF (TODO.md) -- until then, explanations must clearly disclose that the
citation is a paraphrase, not a quote.
"""
from __future__ import annotations

IFAB_EXCERPTS = [
    {
        "id": "law12_direct_free_kick_offenses",
        "law": "Law 12",
        "title": "Fouls and Misconduct -- direct free kick offenses (paraphrased)",
        "text": (
            "A direct free kick is awarded if a player commits any of the following "
            "offenses against an opponent in a manner considered by the referee to be "
            "careless, reckless, or using excessive force: kicking or attempting to "
            "kick, tripping or attempting to trip, jumping at, charging, striking or "
            "attempting to strike, pushing, tackling or challenging."
        ),
    },
    {
        "id": "law12_careless_reckless_excessive",
        "law": "Law 12",
        "title": "Fouls and Misconduct -- careless, reckless, excessive force (paraphrased)",
        "text": (
            "Careless means the player showed a lack of attention or consideration when "
            "making a challenge, or acted without precaution -- no further sanction is "
            "needed beyond the free kick. Reckless means the player acted with disregard "
            "for the danger to an opponent and must be cautioned (yellow card). Excessive "
            "force means the player used far more force than was necessary and must be "
            "sent off (red card)."
        ),
    },
    {
        "id": "law12_advantage",
        "law": "Law 12",
        "title": "Fouls and Misconduct -- advantage (paraphrased)",
        "text": (
            "The referee allows play to continue when the team against which an offense "
            "has been committed will benefit from the advantage, and penalizes the "
            "original offense if the anticipated advantage does not ensue at that time."
        ),
    },
    {
        "id": "law5_referee_decisions",
        "law": "Law 5",
        "title": "The Referee -- decisions and review (paraphrased)",
        "text": (
            "Decisions are made to the best of the referee's ability based on the Laws "
            "of the Game and the spirit of the game, and are based on the observations "
            "of the referee or other match officials. A decision may be reconsidered or "
            "changed based on information from other match officials, provided the "
            "referee has not restarted play or ended the match."
        ),
    },
]
