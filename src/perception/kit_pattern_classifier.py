"""Solid-kit-vs-patterned-kit classifier (2026-07-19): a hard domain
constraint for TeamColorAnchor's referee matching -- real referees never
wear striped/patterned kits (IFAB Law 4), so a crop confidently showing a
patterned kit should never be classified as referee, regardless of its
color-distance to the referee reference centroid.

Built after 5 different cheap statistical proxies (color variance across
horizontal bins, 2-cluster color separation, zero-crossing oscillation
frequency, temporal color-instability across a track's own life, vertical-
edge-gradient density) all failed to reliably separate genuine referees
from striped-kit players on this project's real footage -- in every case,
genuine referee crops scored in the same range as confirmed striped-kit
crops, so no fixed threshold could close the gap without also risking
excluding real referees. Reuses the same frozen-backbone embedding
approach already built and validated for `player_classifier.py`
(MobileNetV3-Small + a shallow scikit-learn head) instead -- see
notebooks/train_kit_pattern_classifier.py for the training data and
validation methodology.

Same graceful-fallback pattern as player_classifier.py: if
weights/kit_pattern_classifier.pkl doesn't exist in this checkout,
classify_boxes returns None and callers treat that as "no constraint
available," not a failure.
"""
from __future__ import annotations

from pathlib import Path

import joblib

from src.perception.player_classifier import embed_crops

WEIGHTS_PATH = Path("weights/kit_pattern_classifier.pkl")
# Only veto a referee match when confident -- see this module's docstring
# for why a plain 0.5 decision boundary isn't used blindly here either;
# calibrated against the held-out validation in
# notebooks/train_kit_pattern_classifier.py.
PATTERNED_CONFIDENCE_THRESHOLD = 0.6

_head = None
_head_checked = False


def _get_head():
    global _head, _head_checked
    if not _head_checked:
        _head = joblib.load(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else None
        _head_checked = True
    return _head


def classify_boxes(frame_bgr, boxes: list[tuple]) -> list[bool] | None:
    """True = confidently a patterned/striped kit (must never be
    classified as referee). Returns None (no-op) if the classifier
    hasn't been trained in this checkout."""
    head = _get_head()
    if head is None or not boxes:
        return None if head is None else []
    embeddings = embed_crops(frame_bgr, boxes)
    proba_patterned = head.predict_proba(embeddings)[:, list(head.classes_).index(1)]
    return [bool(p >= PATTERNED_CONFIDENCE_THRESHOLD) for p in proba_patterned]
