"""분사 조건 -> 침투 깊이 예측, 목표 깊이 -> 구동 조건 역산."""

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from .synth import JetParams, synthesize_shot
from .track import JetTracker

FEATURES = ["v0", "diameter_um", "tau_us"]


def build_dataset(base: JetParams, n: int = 150, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        p = replace(
            base,
            v0=float(rng.uniform(80, 250)),
            diameter_um=float(rng.uniform(80, 200)),
            tau_us=float(rng.uniform(20, 70)),
            seed=seed * 10_000 + k,
        )
        frames, truth = synthesize_shot(p)
        m = JetTracker(p.um_per_px, p.fps).track(frames, nozzle_y=p.nozzle_y)
        rows.append({**{f: getattr(p, f) for f in FEATURES},
                     "peak_velocity_mps": m.peak_velocity_mps,
                     "depth_um": m.final_depth_um})
    return pd.DataFrame(rows).dropna()


def fit_depth_model(df: pd.DataFrame):
    X, y = df[FEATURES].values, df["depth_um"].values
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
    model.fit(X, y)
    return model, float(cv_r2)


def solve_v0_for_depth(model, target_depth_um: float, diameter_um: float, tau_us: float,
                       v_range=(80.0, 250.0), n_grid: int = 400) -> float:
    """다른 조건 고정 시 목표 깊이를 내는 v0 를 격자 탐색으로 역산."""
    vs = np.linspace(*v_range, n_grid)
    X = np.column_stack([vs, np.full_like(vs, diameter_um), np.full_like(vs, tau_us)])
    pred = model.predict(X)
    return float(vs[np.argmin(np.abs(pred - target_depth_um))])
