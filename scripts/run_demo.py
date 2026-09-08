"""엔드투엔드 데모: 합성 영상 -> 추적 -> 재현성 리포트 -> 깊이 예측 모델."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jetlab import JetParams, synthesize_shot, JetTracker
from jetlab.report import run_repeatability, summarize, save_report
from jetlab.predict import build_dataset, fit_depth_model, solve_v0_for_depth


def main():
    out = Path("outputs")
    base = JetParams()

    frames, truth = synthesize_shot(base)
    m = JetTracker(base.um_per_px, base.fps).track(frames, nozzle_y=base.nozzle_y)
    print(f"[single shot] surface_y truth={base.surface_y} detected={m.surface_y}")
    print(f"[single shot] peak velocity {m.peak_velocity_mps:.1f} m/s (v0={base.v0})")
    print(f"[single shot] depth measured {m.final_depth_um:.0f} um / truth {truth['final_depth_um']:.0f} um")
    print(f"[single shot] jet width {m.jet_width_um:.0f} um (nozzle {base.diameter_um} um)")

    df = run_repeatability(base, n_shots=20)
    print("\n[repeatability]\n", summarize(df).round(2))
    save_report(df, out)
    print(f"saved -> {out/'shots.csv'}, {out/'summary.csv'}, {out/'repeatability.png'}")

    ds = build_dataset(base, n=120)
    model, r2 = fit_depth_model(ds)
    print(f"\n[depth model] 5-fold CV R^2 = {r2:.3f} on {len(ds)} synthetic shots")
    v = solve_v0_for_depth(model, target_depth_um=4000, diameter_um=120, tau_us=40)
    print(f"[inverse] target 4000 um at d=120um, tau=40us -> v0 ≈ {v:.0f} m/s")


if __name__ == "__main__":
    main()
