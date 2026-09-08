import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jetlab import JetParams, synthesize_shot, JetTracker


def test_tracker_recovers_surface_and_depth():
    p = JetParams(seed=3)
    frames, truth = synthesize_shot(p)
    m = JetTracker(p.um_per_px, p.fps).track(frames, nozzle_y=p.nozzle_y)

    assert abs(m.surface_y - p.surface_y) <= 2
    # 최종 침투 깊이 오차 2픽셀 이내
    assert abs(m.final_depth_um - truth["final_depth_um"]) <= 2 * p.um_per_px
    # 공기 중 최고 속도는 v0 의 ±25% 이내
    assert 0.75 * p.v0 <= m.peak_velocity_mps <= 1.25 * p.v0


def test_depth_scales_with_v0():
    depths = []
    for v0 in (100.0, 200.0):
        p = JetParams(v0=v0, seed=1)
        frames, _ = synthesize_shot(p)
        m = JetTracker(p.um_per_px, p.fps).track(frames, nozzle_y=p.nozzle_y)
        depths.append(m.final_depth_um)
    assert depths[1] > depths[0] * 1.5
