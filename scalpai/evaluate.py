"""증상별 평가 지표."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from .labels import NUM_SEVERITY, SYMPTOMS


def per_symptom_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """y_true, y_pred: (N, 6). 결측(-1) 은 해당 증상에서 제외.

    지표:
      acc      정확도
      within1  |pred-true| <= 1 비율 (순서형이라 한 등급 차이는 임상적으로 덜 치명적)
      macro_f1 4등급 macro F1
      qwk      quadratic weighted kappa (순서형 일치도, 두피/피부 등급 논문에서 표준)
      n        평가 샘플 수
    """
    rows = []
    for k, s in enumerate(SYMPTOMS):
        m = y_true[:, k] >= 0
        t, p = y_true[m, k], y_pred[m, k]
        if len(t) == 0:
            rows.append({"symptom": s, "n": 0}); continue
        labels = list(range(NUM_SEVERITY))
        rows.append({
            "symptom": s,
            "n": int(len(t)),
            "acc": float((t == p).mean()),
            "within1": float((np.abs(t - p) <= 1).mean()),
            "macro_f1": float(f1_score(t, p, labels=labels, average="macro", zero_division=0)),
            "qwk": float(cohen_kappa_score(t, p, labels=labels, weights="quadratic")) if len(np.unique(t)) > 1 else float("nan"),
        })
    return pd.DataFrame(rows).set_index("symptom")


def confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for k, s in enumerate(SYMPTOMS):
        m = y_true[:, k] >= 0
        out[s] = confusion_matrix(y_true[m, k], y_pred[m, k], labels=list(range(NUM_SEVERITY)))
    return out


def majority_baseline(y_train: np.ndarray, y_eval: np.ndarray) -> pd.DataFrame:
    """학습셋 최빈 등급을 항상 예측하는 베이스라인. 모델이 이보다 나아야 의미가 있다."""
    pred = np.zeros_like(y_eval)
    for k in range(y_train.shape[1]):
        col = y_train[:, k]
        col = col[col >= 0]
        mode = np.bincount(col, minlength=NUM_SEVERITY).argmax() if len(col) else 0
        pred[:, k] = mode
    return per_symptom_metrics(y_eval, pred)
