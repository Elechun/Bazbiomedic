"""다중 샷 재현성 리포트."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .synth import JetParams, synthesize_shot
from .track import JetTracker


def run_repeatability(base: JetParams, n_shots: int = 20, v0_jitter: float = 0.05,
                      tau_jitter: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """구동 조건에 작은 편차를 주어 n_shots 회 분사하고 샷별 지표를 표로 반환."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_shots):
        p = replace(
            base,
            v0=base.v0 * (1 + rng.normal(0, v0_jitter)),
            tau_us=base.tau_us * (1 + rng.normal(0, tau_jitter)),
            seed=seed * 1000 + k,
        )
        frames, truth = synthesize_shot(p)
        m = JetTracker(p.um_per_px, p.fps).track(frames, nozzle_y=p.nozzle_y)
        rows.append({
            "shot": k,
            "v0_true_mps": p.v0,
            "peak_velocity_mps": m.peak_velocity_mps,
            "depth_true_um": truth["final_depth_um"],
            "depth_measured_um": m.final_depth_um,
            "jet_width_um": m.jet_width_um,
            "surface_y_detected": m.surface_y,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["peak_velocity_mps", "depth_measured_um", "jet_width_um"]
    s = df[cols].agg(["mean", "std"]).T
    s["cv_percent"] = 100 * s["std"] / s["mean"]
    s["depth_abs_err_um"] = np.nan
    s.loc["depth_measured_um", "depth_abs_err_um"] = (
        (df["depth_measured_um"] - df["depth_true_um"]).abs().mean()
    )
    return s


def save_report(df: pd.DataFrame, out_dir: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "shots.csv", index=False)
    summarize(df).to_csv(out / "summary.csv")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(df["shot"], df["depth_true_um"], "o-", label="truth")
    ax[0].plot(df["shot"], df["depth_measured_um"], "x--", label="measured")
    ax[0].set_xlabel("shot"); ax[0].set_ylabel("penetration depth [um]"); ax[0].legend()
    ax[1].scatter(df["v0_true_mps"], df["depth_measured_um"])
    ax[1].set_xlabel("true v0 [m/s]"); ax[1].set_ylabel("measured depth [um]")
    fig.tight_layout()
    fig.savefig(out / "repeatability.png", dpi=120)
    plt.close(fig)
    return out
