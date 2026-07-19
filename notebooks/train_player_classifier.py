"""Trains the appearance-based player/non-player classifier (2026-07-18),
scoped in the same session's PROGRESS.md entry: a lightweight, frozen-
backbone + shallow-head classifier, using this project's OWN already-
collected `player`/`non_player` crops as training data -- zero new manual
annotation, since Layer 1's existing `cls` field already provides an
auto-labeled source (see the scoping entry for why this labeling is
trustworthy: `player` on a reliable `calib_source=="own"` shot, `non_player`
on a confidently-outside-the-pitch reliable-calibration detection).

Deliberately NOT full re-identification -- see player_classifier.py's
module docstring for the reasoning. Train/val split is by CLIP, not by
frame, so the same track's many near-duplicate frames can't leak across
the split and inflate the validation numbers.

Licensing note, same as every other real-clip artifact in this project:
the extracted crops are derived from real broadcast footage under
`data/raw/soccernet/` (gitignored) -- `data/processed/player_classifier/`
and `weights/player_classifier.pkl` are both gitignored too, local-only,
never committed or distributed.

Usage:
    python -m notebooks.train_player_classifier prepare   # extract+save crops
    python -m notebooks.train_player_classifier train     # fit the LR head
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

CROPS_DIR = Path("data/processed/player_classifier")
WEIGHTS_OUT = Path("weights/player_classifier.pkl")
MAX_POS_PER_CLIP = 300  # caps how many near-duplicate player crops one clip can contribute
RANDOM_SEED = 0

CLIPS = [
    "data/raw/soccernet/card_chelsea_burnley.mp4",
    "data/raw/soccernet/card_hull_arsenal.mp4",
    "data/raw/soccernet/card_mancity_watford.mp4",
    "data/raw/soccernet/card_palace_arsenal.mp4",
    "data/raw/soccernet/card_southampton_liverpool.mp4",
    "data/raw/soccernet/card_swansea_manutd.mp4",
    "data/raw/soccernet/clip20s.mp4",
    "data/raw/soccernet/foul_arsenal_anderlecht.mp4",
    "data/raw/soccernet/foul_before_goal_clip.mp4",
    "data/raw/soccernet/foul_leicester_mancity.mp4",
]
# Chelsea-Burnley is deliberately EXCLUDED from training data entirely (not
# just held out for validation): it has zero auto-labeled non_player rows
# of its own (its crowd is on fallback_prev_shot shots, so the calibration-
# based check never runs there), which makes it a genuine, uncontaminated
# out-of-training-distribution test for the appearance classifier -- see
# the validation entry in PROGRESS.md.
CLIPS = [c for c in CLIPS if "chelsea_burnley" not in c]
VAL_CLIP_STEM = "foul_leicester_mancity"  # held out entirely for the quantitative split


def _clip_stem(path: str) -> str:
    return Path(path).stem


def prepare() -> None:
    from src.perception.pipeline import run_perception

    rng = np.random.default_rng(RANDOM_SEED)
    for split in ("positive", "negative"):
        (CROPS_DIR / split).mkdir(parents=True, exist_ok=True)

    for clip_path in CLIPS:
        stem = _clip_stem(clip_path)
        print(f"--- {stem} ---")
        df = run_perception(clip_path, backend="yolo")

        pos_rows = df[(df["cls"] == "player") & (df["calib_source"] == "own")]
        neg_rows = df[df["cls"] == "non_player"]
        if len(pos_rows) > MAX_POS_PER_CLIP:
            pos_rows = pos_rows.iloc[rng.choice(len(pos_rows), MAX_POS_PER_CLIP, replace=False)]
        print(f"  positive candidates: {len(pos_rows)} (of {(df['cls'] == 'player').sum()} total player rows)")
        print(f"  negative candidates: {len(neg_rows)}")

        wanted = {}  # frame -> list of (split, track_id, box)
        for split, rows in (("positive", pos_rows), ("negative", neg_rows)):
            for _, r in rows.iterrows():
                wanted.setdefault(int(r["frame"]), []).append(
                    (split, int(r["track_id"]), (r["box_x1"], r["box_y1"], r["box_x2"], r["box_y2"]))
                )
        if not wanted:
            continue

        cap = cv2.VideoCapture(clip_path)
        frame_idx = 0
        max_frame = max(wanted)
        n_saved = 0
        while frame_idx <= max_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in wanted:
                for split, track_id, (x1, y1, x2, y2) in wanted[frame_idx]:
                    x1i, y1i = max(0, int(x1)), max(0, int(y1))
                    x2i, y2i = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
                    if x2i <= x1i or y2i <= y1i:
                        continue
                    crop = frame[y1i:y2i, x1i:x2i]
                    out_path = CROPS_DIR / split / f"{stem}_{frame_idx}_{track_id}.jpg"
                    cv2.imwrite(str(out_path), crop)
                    n_saved += 1
            frame_idx += 1
        cap.release()
        print(f"  saved {n_saved} crops")


def _load_embeddings(paths: list[Path]) -> np.ndarray:
    import torch

    from src.perception.player_classifier import _get_backbone

    embeddings = []
    batch_size = 32
    for i in range(0, len(paths), batch_size):
        batch = np.stack([_crop_to_input(cv2.imread(str(p))) for p in paths[i:i + batch_size]])
        with torch.no_grad():
            out = _get_backbone()(torch.from_numpy(batch).float())
        embeddings.append(out.numpy())
    return np.concatenate(embeddings, axis=0)


def _crop_to_input(crop_bgr: np.ndarray) -> np.ndarray:
    from src.perception.player_classifier import CROP_SIZE, _IMAGENET_MEAN, _IMAGENET_STD

    resized = cv2.resize(crop_bgr, (CROP_SIZE, CROP_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return normalized.transpose(2, 0, 1)


def train() -> None:
    records = []
    for split, label in (("positive", 1), ("negative", 0)):
        for p in sorted((CROPS_DIR / split).glob("*.jpg")):
            stem = p.stem.rsplit("_", 2)[0]  # strip _<frame>_<track>
            records.append({"path": p, "label": label, "clip_stem": stem})
    meta = pd.DataFrame(records)
    print(f"Total crops: {len(meta)} ({(meta['label'] == 1).sum()} positive, {(meta['label'] == 0).sum()} negative)")
    print(meta.groupby(["clip_stem", "label"]).size())

    val_mask = meta["clip_stem"] == VAL_CLIP_STEM
    train_meta, val_meta = meta[~val_mask], meta[val_mask]
    print(f"Train: {len(train_meta)} crops from {train_meta['clip_stem'].nunique()} clips. "
          f"Val (held out clip '{VAL_CLIP_STEM}'): {len(val_meta)} crops.")

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
        print(f"  predicted positive (player-like): {int(val_pred.sum())}/{len(val_pred)}")

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
