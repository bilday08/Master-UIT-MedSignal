# [M2] Nhanh Tabular: MLP encode du lieu lam sang -> feature vector.
from __future__ import annotations

import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    """
    Input: [B, in_dim] (8 numeric da scale + Sex = 9 feature).
    Output: [B, feat_dim] feature vector cho lop Fusion.

    M2:
      - Experiment with norm="batch" (default) vs "layer" (Streamlit demo).
      - Add embedding for feature Sex.
    """

    def __init__(
        self,
        in_dim: int = 9,
        hidden: tuple = (64, 32),
        feat_dim: int = 32,
        dropout: float = 0.3,
        norm: str = "batch",
        sex_embed_dim: int = 0,
    ):
        super().__init__()
        assert norm in ("batch", "layer"), f"norm phai la 'batch' hoac 'layer', nhan '{norm}'"

        self.sex_embed_dim = sex_embed_dim
        if sex_embed_dim > 0:
            # Sex truyen vao la LongTensor index (0=Female, 1=Male)
            self.sex_embed = nn.Embedding(2, sex_embed_dim)
            first_in = in_dim + sex_embed_dim  # 8 numeric + embed
        else:
            self.sex_embed = None
            first_in = in_dim  # 9 = 8 numeric + Sex float

        dims = [first_in, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            norm_layer = nn.BatchNorm1d(b) if norm == "batch" else nn.LayerNorm(b)
            layers += [nn.Linear(a, b), norm_layer, nn.ReLU(), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], feat_dim)

    def forward(self, x: torch.Tensor, sex_idx: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, in_dim] — neu sex_embed_dim=0 thi x gom ca Sex float (in_dim=9).
        sex_idx  : [B] LongTensor chi dung khi sex_embed_dim > 0.
        """
        if self.sex_embed is not None:
            assert sex_idx is not None, "sex_idx bat buoc khi sex_embed_dim > 0"
            emb = self.sex_embed(sex_idx)          # [B, sex_embed_dim]
            x = torch.cat([x, emb], dim=1)         # [B, 8 + sex_embed_dim]
        return self.head(self.backbone(x))


class TabularClassifier(nn.Module):
    """
    Baseline tabular-only cho task Plaque_present — MLP + 1 logit.
    Dung lam moc so sanh voi multimodal.

    norm="layer" de an toan khi inference don le (Streamlit demo).
    """

    def __init__(
        self,
        in_dim: int = 9,
        hidden: tuple = (64, 32),
        dropout: float = 0.3,
        norm: str = "layer",
        sex_embed_dim: int = 0,
    ):
        super().__init__()
        self.encoder = TabularMLP(
            in_dim=in_dim, hidden=hidden, feat_dim=hidden[-1],
            dropout=dropout, norm=norm, sex_embed_dim=sex_embed_dim,
        )
        self.classifier = nn.Linear(hidden[-1], 1)

    def forward(self, x: torch.Tensor, sex_idx: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(self.encoder(x, sex_idx))  # [B,1] logit
