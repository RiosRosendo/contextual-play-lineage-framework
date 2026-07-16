"""Two-branch foul classifier, per the project spec section 4: a video encoder
(contact biomechanics) + a sequence encoder (trajectory features), fused
before classification.

Skeleton-phase simplification (documented in the internal task list, not a silent
shortcut): a real video encoder (VideoMAE/X3D) needs a pretrained checkpoint
and fine-tuning data we don't have yet, so the "video branch" here is a
small linear head over the motion-intensity summary from
features.extract_video_branch_features -- a numeric proxy for what a video
encoder would extract from pixels around contact. The sequence branch is a
real (but untrained) LSTM over trajectory features. Weights are seeded for
determinism; outputs are illustrative until trained on SoccerNet-MVFouls.
"""
from __future__ import annotations

import torch
import torch.nn as nn

SEQ_FEATURE_DIM = 6
VIDEO_FEATURE_DIM = 4
EMBED_DIM = 16


class TwoBranchFoulClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence_encoder = nn.LSTM(SEQ_FEATURE_DIM, EMBED_DIM, batch_first=True)
        self.video_encoder = nn.Sequential(nn.Linear(VIDEO_FEATURE_DIM, EMBED_DIM), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, EMBED_DIM), nn.ReLU(), nn.Linear(EMBED_DIM, 1),
        )

    def forward(self, seq_features: torch.Tensor, video_features: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.sequence_encoder(seq_features)
        seq_embed = h_n[-1]
        video_embed = self.video_encoder(video_features)
        fused = torch.cat([seq_embed, video_embed], dim=-1)
        return torch.sigmoid(self.fusion(fused)).squeeze(-1)


_model = None


def get_model() -> TwoBranchFoulClassifier:
    global _model
    if _model is None:
        torch.manual_seed(42)
        _model = TwoBranchFoulClassifier()
        _model.eval()
    return _model
