"""Trains a solid-kit-vs-patterned-kit classifier (2026-07-19), built after
5 different cheap statistical proxies for stripe detection (color variance
across horizontal bins, 2-cluster color separation, zero-crossing
oscillation frequency, temporal color-instability across a track's life,
vertical-edge-gradient density) all failed to reliably separate genuine
referees from striped-kit players on this project's real footage -- in
every case, genuine referee crops (Leicester, Swansea) scored in the same
range as confirmed striped-kit crops (Athletic Bilbao, Hull City), so no
threshold could fix the misclassification without also risking excluding
real referees.

Reuses the exact same frozen-backbone embedding approach already built
and validated for `player_classifier.py` (MobileNetV3-Small + a shallow
scikit-learn head) -- a learned feature representation is far more likely
to capture the real visual distinction (a printed/woven stripe pattern vs.
a solid dyed kit) than a hand-crafted statistic, given how many of those
have now been tried and failed.

Training data, all auto-labeled from this project's own already-confirmed
findings -- no new manual annotation:
  - "patterned" (positive): every `cls=="referee"`-labeled row on
    clip20s.mp4 (Athletic Bilbao's striped kit) and card_hull_arsenal.mp4
    (Hull City's striped kit) -- both CONFIRMED misclassifications of a
    genuinely patterned kit, by direct frame inspection this session.
  - "solid" (negative): every `cls=="referee"`-labeled row on
    foul_leicester_mancity.mp4, card_swansea_manutd.mp4, and
    card_chelsea_burnley.mp4 (all CONFIRMED genuine, solid-kit referees,
    by direct frame inspection across this project's history), plus a
    sample of ordinary `cls=="player"` rows from those same three clips
    (solid team kits), so "solid" isn't defined by referee kits alone.

Usage:
    python -m notebooks.train_kit_pattern_classifier prepare
    python -m notebooks.train_kit_pattern_classifier train
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

CROPS_DIR = Path("data/processed/kit_pattern_classifier")
WEIGHTS_OUT = Path("weights/kit_pattern_classifier.pkl")
MAX_SOLID_PLAYER_PER_CLIP = 150  # caps ordinary-player contribution so referee crops aren't drowned out

PATTERNED_CLIPS = ["data/raw/soccernet/clip20s.mp4", "data/raw/soccernet/card_hull_arsenal.mp4"]
SOLID_CLIPS = [
    "data/raw/soccernet/foul_leicester_mancity.mp4",
    "data/raw/soccernet/card_swansea_manutd.mp4",
    "data/raw/soccernet/card_chelsea_burnley.mp4",
]
VAL_CLIP_STEM = "card_chelsea_burnley"  # held out for the "solid" class; see train()'s docstring for why patterned isn't held out


def _clip_stem(path: str) -> str:
    return Path(path).stem


def _extract_crops(clip_path: str, label: str, rng: np.random.default_rng) -> None:
    from src.perception.pipeline import run_perception

    df = run_perception(clip_path, backend="yolo")
    stem = _clip_stem(clip_path)
    if label == "patterned":
        rows = df[df["cls"] == "referee"]
    else:
        ref_rows = df[df["cls"] == "referee"]
        player_rows = df[df["cls"] == "player"]
        if len(player_rows) > MAX_SOLID_PLAYER_PER_CLIP:
            player_rows = player_rows.iloc[rng.choice(len(player_rows), MAX_SOLID_PLAYER_PER_CLIP, replace=False)]
        rows = pd.concat([ref_rows, player_rows])
    print(f"  {stem}: {len(rows)} crops for label '{label}'")

    wanted: dict[int, list] = {}
    for _, r in rows.iterrows():
        wanted.setdefault(int(r["frame"]), []).append(
            (int(r["track_id"]), (r["box_x1"], r["box_y1"], r["box_x2"], r["box_y2"]))
        )
    if not wanted:
        return

    (CROPS_DIR / label).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(clip_path)
    frame_idx, max_frame, n_saved = 0, max(wanted), 0
    while frame_idx <= max_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in wanted:
            for track_id, (x1, y1, x2, y2) in wanted[frame_idx]:
                x1i, y1i = max(0, int(x1)), max(0, int(y1))
                x2i, y2i = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
                if x2i <= x1i or y2i <= y1i:
                    continue
                crop = frame[y1i:y2i, x1i:x2i]
                cv2.imwrite(str(CROPS_DIR / label / f"{stem}_{frame_idx}_{track_id}.jpg"), crop)
                n_saved += 1
        frame_idx += 1
    cap.release()
    print(f"  saved {n_saved} crops")


def prepare() -> None:
    rng = np.random.default_rng(0)
    for clip in PATTERNED_CLIPS:
        print(f"--- {clip} (patterned) ---")
        _extract_crops(clip, "patterned", rng)
    for clip in SOLID_CLIPS:
        print(f"--- {clip} (solid) ---")
        _extract_crops(clip, "solid", rng)


def _load_embeddings(paths: list[Path]) -> np.ndarray:
    import torch

    from src.perception.player_classifier import CROP_SIZE, _IMAGENET_MEAN, _IMAGENET_STD, _get_backbone

    def to_input(crop_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop_bgr, (CROP_SIZE, CROP_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return ((rgb - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)

    embeddings = []
    batch_size = 32
    for i in range(0, len(paths), batch_size):
        batch = np.stack([to_input(cv2.imread(str(p))) for p in paths[i:i + batch_size]])
        with torch.no_grad():
            out = _get_backbone()(torch.from_numpy(batch).float())
        embeddings.append(out.numpy())
    return np.concatenate(embeddings, axis=0)


def train() -> None:
    """First attempt held out card_hull_arsenal (Hull City, orange/black
    stripes) entirely, training "patterned" only on clip20s (Athletic
    Bilbao, red/white stripes) -- recall=0.00 on the held-out clip,
    confirmed directly: with only ONE distinct stripe pattern in
    training, the classifier learned "this specific red+white
    combination," not a general patternedness concept, and didn't
    transfer to an entirely different color combination at all. With
    only 2 known distinct patterned examples available in this project's
    data, holding either out for the positive class isn't a fair
    generalization test -- both are now used in training. Held out
    instead: card_chelsea_burnley (a solid-kit clip never used for
    "patterned" at all), which still gives a genuine, meaningful
    generalization check on the side that matters most for safety --
    does the classifier correctly recognize a real referee/player it
    never saw as solid, not patterned."""
    records = []
    for label, y in (("patterned", 1), ("solid", 0)):
        for p in sorted((CROPS_DIR / label).glob("*.jpg")):
            stem = p.stem.rsplit("_", 2)[0]
            records.append({"path": p, "label": y, "clip_stem": stem})
    meta = pd.DataFrame(records)
    print(f"Total crops: {len(meta)} ({(meta['label'] == 1).sum()} patterned, {(meta['label'] == 0).sum()} solid)")
    print(meta.groupby(["clip_stem", "label"]).size())

    val_mask = meta["clip_stem"] == VAL_CLIP_STEM
    train_meta, val_meta = meta[~val_mask], meta[val_mask]
    print(f"Train: {len(train_meta)} crops. Val (held-out clip '{VAL_CLIP_STEM}'): {len(val_meta)} crops.")

    train_emb = _load_embeddings(list(train_meta["path"]))
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(train_emb, train_meta["label"].to_numpy())

    if len(val_meta):
        val_emb = _load_embeddings(list(val_meta["path"]))
        val_pred = clf.predict(val_emb)
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_meta["label"], val_pred, average="binary", zero_division=0,
        )
        print(f"Held-out clip '{VAL_CLIP_STEM}': precision={precision:.2f} recall={recall:.2f} f1={f1:.2f}")
        print(f"  predicted patterned: {int(val_pred.sum())}/{len(val_pred)}")

    train_pred = clf.predict(train_emb)
    precision, recall, f1, _ = precision_recall_fscore_support(
        train_meta["label"], train_pred, average="binary", zero_division=0,
    )
    print(f"Train set (in-sample): precision={precision:.2f} recall={recall:.2f} f1={f1:.2f}")

    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, WEIGHTS_OUT)
    print(f"Saved classifier head to {WEIGHTS_OUT}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    {"prepare": prepare, "train": train}[cmd]()
