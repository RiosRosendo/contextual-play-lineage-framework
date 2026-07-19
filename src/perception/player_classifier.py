"""Appearance-based player/non-player classifier (2026-07-18): a first-stage
filter ahead of TeamColorAnchor, deliberately decoupled from calibration.

Motivation: this session's contact-confirmation work found a real, residual
false positive (Swansea-Man Utd, t=3.16s) that no proximity/calibration fix
can close -- a spectator's own keypoints happened to land pixel-close to a
real player's, on a shot whose calibration is unreliable so the existing
position-based non_player check can't run there either (see PROGRESS.md).
That gap is specifically "is this crop a real athlete at all," a question
appearance alone can answer independent of calibrated position or shot
quality -- which the existing non_player check structurally cannot.

Deliberately NOT full re-identification: no triplet/contrastive loss, no
cross-frame identity matching, no large embedding network. A frozen
ImageNet-pretrained MobileNetV3-Small backbone (already a torchvision
dependency-free download, no licensing concern the way SoccerSum-derived
ball-detector weights have) produces a 576-dim embedding per crop; a small
logistic-regression head (scikit-learn), trained on this project's own
already-collected `player`/`non_player` crops, does the binary
player-or-referee-appearance vs. non-player classification on top of it.

Graceful fallback, same pattern as yolo_detector.py's fine-tuned-weights
check: if `weights/player_classifier.pkl` doesn't exist (not yet trained,
e.g. a fresh checkout), `classify_boxes` returns None and every caller
treats that as "no appearance filtering available" rather than failing.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

WEIGHTS_PATH = Path("weights/player_classifier.pkl")
CROP_SIZE = 224
NON_PLAYER_CONFIDENCE_THRESHOLD = 0.9  # see classify_boxes' docstring
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_backbone = None
_head = None
_head_checked = False


def _get_backbone() -> torch.nn.Module:
    global _backbone
    if _backbone is None:
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Identity()  # keep avgpool's 576-dim flatten, drop the ImageNet head
        model.eval()
        _backbone = model
    return _backbone


def _get_head():
    """Lazily loads the trained logistic-regression head. Cached at module
    level -- checked once per process, not once per frame."""
    global _head, _head_checked
    if not _head_checked:
        _head = joblib.load(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else None
        _head_checked = True
    return _head


def _preprocess_crop(frame_bgr: np.ndarray, box: tuple) -> np.ndarray:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        crop = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(frame_bgr[y1:y2, x1:x2], (CROP_SIZE, CROP_SIZE))
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return normalized.transpose(2, 0, 1)  # HWC -> CHW


def embed_crops(frame_bgr: np.ndarray, boxes: list[tuple]) -> np.ndarray:
    """Returns an (N, 576) embedding array, one row per box, batched through
    the frozen backbone in a single forward pass (not N sequential ones --
    matters for keeping full-clip runtime reasonable, see the dev log's
    per-clip timing note)."""
    if not boxes:
        return np.empty((0, 576), dtype=np.float32)
    batch = np.stack([_preprocess_crop(frame_bgr, b) for b in boxes])
    with torch.no_grad():
        out = _get_backbone()(torch.from_numpy(batch).float())
    return out.numpy()


def classify_boxes(frame_bgr: np.ndarray, boxes: list[tuple]) -> list[bool] | None:
    """True = player/referee-like appearance, False = non-player (crowd/
    bench/staff). Returns None (no-op -- caller should treat everyone as
    passing) if the classifier hasn't been trained yet in this checkout.

    Uses `NON_PLAYER_CONFIDENCE_THRESHOLD` rather than the model's default
    0.5 decision boundary (2026-07-18): real-footage validation on
    Chelsea-Burnley -- deliberately excluded from training, see
    train_player_classifier.py -- found the default boundary produces a
    real, disclosed regression: motion-blurred crops during an actual
    tackle (visually atypical vs. the training data's more ordinary open-
    play crops) get confidently misclassified as non_player, at the exact
    moment a foul candidate most needs its participants to still read as
    players. Only exclude a crop when the model is VERY confident it's
    NOT a player (P(non_player) >= 0.9) -- ambiguous/borderline cases fall
    through as "player," trading some missed crowd exclusions for not
    losing real, foul-relevant contact. Not a full fix -- see PROGRESS.md
    for the still-open question of whether this is sufficient."""
    head = _get_head()
    if head is None or not boxes:
        return None if head is None else []
    embeddings = embed_crops(frame_bgr, boxes)
    proba_player = head.predict_proba(embeddings)[:, list(head.classes_).index(1)]
    return [bool(p >= 1 - NON_PLAYER_CONFIDENCE_THRESHOLD) for p in proba_player]
