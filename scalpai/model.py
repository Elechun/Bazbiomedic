"""공유 백본 + 증상별 헤드 다중 태스크 분류기."""

from __future__ import annotations

import torch
import torch.nn as nn

from .labels import NUM_SEVERITY, SYMPTOMS


class TinyCNN(nn.Module):
    """CPU 스모크 테스트용 소형 백본."""

    def __init__(self, width: int = 16):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, width), block(width, width * 2), block(width * 2, width * 4),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.out_dim = width * 4

    def forward(self, x):
        return self.features(x)


def build_backbone(name: str, pretrained: bool):
    if name == "tiny":
        return TinyCNN()
    import torchvision.models as M
    if name == "efficientnet_b0":
        m = M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        out_dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
    elif name == "resnet18":
        m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        out_dim = m.fc.in_features
        m.fc = nn.Identity()
    else:
        raise ValueError(name)
    m.out_dim = out_dim
    return m


class MultiHeadScalpNet(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained)
        d = self.backbone.out_dim
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({s: nn.Linear(d, NUM_SEVERITY) for s in SYMPTOMS})

    def forward(self, x) -> dict[str, torch.Tensor]:
        f = self.dropout(self.backbone(x))
        return {s: h(f) for s, h in self.heads.items()}


def multitask_loss(logits: dict[str, torch.Tensor], y: torch.Tensor,
                   weights: dict[str, torch.Tensor] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """증상별 CE 의 평균. y[:, k] == -1 (결측) 은 해당 헤드 손실에서 제외."""
    total, parts, n_heads = 0.0, {}, 0
    for k, s in enumerate(SYMPTOMS):
        t = y[:, k]
        m = t >= 0
        if m.sum() == 0:
            continue
        w = weights[s] if weights is not None else None
        l = nn.functional.cross_entropy(logits[s][m], t[m], weight=w)
        total = total + l
        parts[s] = float(l.detach())
        n_heads += 1
    return total / max(n_heads, 1), parts
