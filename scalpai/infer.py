"""단일 이미지 추론과 시술 전후 비교 리포트."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .data import build_transforms
from .labels import SEVERITY_NAMES_KO, SYMPTOMS, SYMPTOM_NAMES_KO
from .model import MultiHeadScalpNet
from .train import TrainConfig


def load_model(checkpoint: str | Path, device: str = "cpu"):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in ck["cfg"].items() if k in TrainConfig.__dataclass_fields__})
    model = MultiHeadScalpNet(cfg.backbone, pretrained=False)
    model.load_state_dict(ck["model"]); model.eval().to(device)
    return model, cfg


@torch.no_grad()
def predict_image(model, cfg, path: str | Path, device: str = "cpu") -> dict[str, dict]:
    x = build_transforms(cfg.image_size, False)(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    out = model(x)
    res = {}
    for s in SYMPTOMS:
        p = torch.softmax(out[s][0], 0).cpu()
        g = int(p.argmax())
        res[s] = {"grade": g, "grade_ko": SEVERITY_NAMES_KO[g], "prob": [round(float(v), 3) for v in p],
                  "expected_grade": round(float((p * torch.arange(4)).sum()), 2)}
    return res


def before_after_report(model, cfg, before: str | Path, after: str | Path, device: str = "cpu") -> str:
    b, a = predict_image(model, cfg, before, device), predict_image(model, cfg, after, device)
    lines = ["증상            | 시술 전      | 시술 후      | 변화(기대등급)"]
    for s in SYMPTOMS:
        d = a[s]["expected_grade"] - b[s]["expected_grade"]
        lines.append(f"{SYMPTOM_NAMES_KO[s]:<8} | {b[s]['grade_ko']:<6}({b[s]['expected_grade']:.2f}) | "
                     f"{a[s]['grade_ko']:<6}({a[s]['expected_grade']:.2f}) | {d:+.2f}")
    lines.append("주의: 모델 출력은 보조 지표이며 임상 판단을 대체하지 않는다.")
    return "\n".join(lines)
