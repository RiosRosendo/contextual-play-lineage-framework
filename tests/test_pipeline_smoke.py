"""Smoke test for the full skeleton pipeline (CLAUDE.md section 3: validate
the whole architecture runs end-to-end before deepening any single layer).
Not a correctness test of any individual model -- just confirms every layer
and module wires together without errors and Module A's review-alert
mechanism fires on the scripted unflagged-foul-before-goal clip.
"""
from __future__ import annotations

from pathlib import Path

from src.perception.synthetic_clip import generate_synthetic_clip
from src.run_pipeline import run_pipeline

CLIP_PATH = "data/raw/synthetic_match_clip.mp4"


def test_full_pipeline_runs_end_to_end():
    if not Path(CLIP_PATH).exists():
        generate_synthetic_clip(CLIP_PATH)

    result = run_pipeline(CLIP_PATH, backend="color")

    assert len(result["perception_df"]) > 0
    assert len(result["events"]) > 0
    assert any(e["type"] == "foul" for e in result["events"])
    assert any(e["type"] == "goal" for e in result["events"])
    assert result["predictive"]["pitch_control_at_goal"] is not None
    assert len(result["review_alerts"]) >= 1
