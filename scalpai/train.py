"""학습 루프."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import ScalpDataset, build_transforms, class_weights
from .evaluate import per_symptom_metrics
from .labels import SYMPTOMS
from .model import MultiHeadScalpNet, multitask_loss


@dataclass
class TrainConfig:
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    class_weight_power: float = 0.5   # 0 이면 가중치 없음
    num_workers: int = 2
    seed: int = 0
    select_metric: str = "qwk"        # 검증 선택 기준: 증상 평균 qwk (또는 macro_f1, acc)
    patience: int = 5
    grad_clip: float = 0.0            # 0 이면 클리핑 없음
    device: str = "auto"


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id: int):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)


def _device(cfg: TrainConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, ps, idx = [], [], []
    for x, y, i in loader:
        out = model(x.to(device))
        p = torch.stack([out[s].argmax(1) for s in SYMPTOMS], dim=1).cpu()
        ys.append(y); ps.append(p); idx.append(i)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(idx).numpy()


def train(df: pd.DataFrame, out_dir: str | Path, cfg: TrainConfig) -> dict:
    """df 는 split 열(train/val/test) 을 가진 인덱스 표. 결과와 로그를 out_dir 에 저장."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = _device(cfg)

    tr = df[df["split"] == "train"]; va = df[df["split"] == "val"]
    assert len(tr) and len(va), "train/val 이 비어 있음"
    assert not set(tr["group"]) & set(va["group"]), "train/val 피험자 누수"

    g = torch.Generator(); g.manual_seed(cfg.seed)
    dl_tr = DataLoader(ScalpDataset(tr, build_transforms(cfg.image_size, True)), batch_size=cfg.batch_size,
                       shuffle=True, num_workers=cfg.num_workers, worker_init_fn=_worker_init, generator=g, drop_last=False)
    dl_va = DataLoader(ScalpDataset(va, build_transforms(cfg.image_size, False)), batch_size=cfg.batch_size,
                       shuffle=False, num_workers=cfg.num_workers)

    model = MultiHeadScalpNet(cfg.backbone, cfg.pretrained).to(device)
    weights = None
    if cfg.class_weight_power > 0:
        weights = {s: torch.tensor(w, dtype=torch.float32, device=device) for s, w in class_weights(tr, cfg.class_weight_power).items()}
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    with open(out / "config.json", "w") as f:
        json.dump({**asdict(cfg), "device": str(device), "n_train": len(tr), "n_val": len(va),
                   "torch": torch.__version__}, f, indent=2, ensure_ascii=False)
    log_path = out / "train_log.csv"
    fields = ["epoch", "time_s", "lr", "train_loss", *[f"loss_{s}" for s in SYMPTOMS], "val_acc", "val_macro_f1", "val_qwk", "val_within1"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best, best_epoch, bad = -np.inf, -1, 0
    for ep in range(cfg.epochs):
        t0 = time.time(); model.train(); tot, n, parts_sum = 0.0, 0, {s: 0.0 for s in SYMPTOMS}
        for x, y, _ in dl_tr:
            x, y = x.to(device), y.to(device)
            loss, parts = multitask_loss(model(x), y, weights)
            opt.zero_grad(set_to_none=True); loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += float(loss) * len(x); n += len(x)
            for s, v in parts.items(): parts_sum[s] += v * len(x)
        sched.step()
        yt, yp, _ = predict(model, dl_va, device)
        met = per_symptom_metrics(yt, yp)
        row = {"epoch": ep, "time_s": round(time.time() - t0, 1), "lr": opt.param_groups[0]["lr"], "train_loss": tot / n,
               **{f"loss_{s}": parts_sum[s] / n for s in SYMPTOMS},
               "val_acc": met["acc"].mean(), "val_macro_f1": met["macro_f1"].mean(),
               "val_qwk": met["qwk"].mean(skipna=True), "val_within1": met["within1"].mean()}
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        score = row[f"val_{cfg.select_metric}"]
        print(f"[ep {ep:02d}] loss {row['train_loss']:.4f} | val acc {row['val_acc']:.3f} f1 {row['val_macro_f1']:.3f} qwk {row['val_qwk']:.3f} | {row['time_s']}s")
        if np.isfinite(score) and score > best:
            best, best_epoch, bad = score, ep, 0
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": ep, "score": float(score)}, out / "best.pt")
            met.to_csv(out / "val_metrics_best.csv")
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"early stop at epoch {ep} (best {best_epoch})"); break
    return {"best_epoch": best_epoch, "best_score": float(best), "log": str(log_path), "checkpoint": str(out / "best.pt")}


def evaluate_split(df: pd.DataFrame, checkpoint: str | Path, split: str = "test", batch_size: int = 32,
                   num_workers: int = 2) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in ck["cfg"].items() if k in TrainConfig.__dataclass_fields__})
    device = _device(cfg)
    model = MultiHeadScalpNet(cfg.backbone, pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    sub = df[df["split"] == split]
    dl = DataLoader(ScalpDataset(sub, build_transforms(cfg.image_size, False)), batch_size=batch_size, num_workers=num_workers)
    yt, yp, _ = predict(model, dl, device)
    return per_symptom_metrics(yt, yp), yt, yp
