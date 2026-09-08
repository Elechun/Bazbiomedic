"""합성 고속 카메라 영상 생성기.

실제 고속 영상 데이터가 없을 때 추적 파이프라인을 검증하기 위한 단순 물리 모델.
- 공기 중: 제트 선단이 등속 v0 로 이동
- 모사체(젤라틴) 진입 후: 속도가 시간상수 tau 로 지수 감쇠
  -> 최종 침투 깊이 = v0 * tau (이론값, 정답 라벨로 사용)
"""

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class JetParams:
    v0: float = 150.0          # 초기 제트 속도 [m/s]
    diameter_um: float = 120.0 # 노즐/제트 지름 [um]
    tau_us: float = 40.0       # 모사체 내 속도 감쇠 시간상수 [us]
    duration_us: float = 400.0 # 분사 지속 시간 [us]
    fps: float = 200_000.0     # 고속 카메라 프레임 속도
    um_per_px: float = 40.0    # 공간 해상도
    height: int = 512
    width: int = 128
    nozzle_y: int = 16         # 노즐 출구 행
    surface_y: int = 96        # 모사체 표면 행
    noise_sigma: float = 6.0   # 센서 노이즈 (8bit 기준)
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _tip_position_m(t_s: np.ndarray, p: JetParams) -> np.ndarray:
    """시간 배열(초)에 대한 선단 위치 [m], 노즐 기준."""
    air_len = (p.surface_y - p.nozzle_y) * p.um_per_px * 1e-6
    t_entry = air_len / p.v0
    tau = p.tau_us * 1e-6
    pos = np.where(
        t_s < t_entry,
        p.v0 * t_s,
        air_len + p.v0 * tau * (1.0 - np.exp(-(t_s - t_entry) / tau)),
    )
    return pos


def synthesize_shot(p: JetParams):
    """한 번의 분사에 대한 (frames, truth) 반환.

    frames: (N, H, W) uint8. 첫 8프레임은 분사 전 배경.
    truth: dict, 프레임별 정답 선단 행 위치와 최종 침투 깊이[um].
    """
    rng = np.random.default_rng(p.seed)
    n_pre = 8
    n_shot = int(p.duration_us * 1e-6 * p.fps)
    n = n_pre + n_shot

    bg = np.full((p.height, p.width), 40.0, dtype=np.float32)
    bg[p.surface_y :, :] = 70.0  # 모사체는 배경보다 약간 밝음
    bg += rng.normal(0, 2.0, bg.shape)  # 고정 패턴

    t = (np.arange(n_shot) + 1) / p.fps
    tip_m = _tip_position_m(t, p)
    tip_px = p.nozzle_y + tip_m / (p.um_per_px * 1e-6)
    tip_px = np.clip(tip_px, p.nozzle_y, p.height - 1)

    r_px = max(1.0, (p.diameter_um / p.um_per_px) / 2.0)
    cx = p.width // 2
    xs = np.arange(p.width)

    frames = np.empty((n, p.height, p.width), dtype=np.uint8)
    for i in range(n):
        img = bg.copy()
        if i >= n_pre:
            ty = tip_px[i - n_pre]
            ys = np.arange(p.nozzle_y, int(np.floor(ty)) + 1)
            # 모사체 안에서는 제트가 약간 퍼짐
            widen = np.where(ys >= p.surface_y, 1.0 + 0.004 * (ys - p.surface_y), 1.0)
            rr = r_px * widen
            dx = (xs[None, :] - cx) ** 2
            prof = np.exp(-dx / (2.0 * (rr[:, None] ** 2)))
            img[ys, :] += 160.0 * prof
        img += rng.normal(0, p.noise_sigma, img.shape)
        frames[i] = np.clip(img, 0, 255).astype(np.uint8)

    depth_um = max(0.0, (tip_px[-1] - p.surface_y) * p.um_per_px)
    truth = {
        "tip_y": np.concatenate([np.full(n_pre, np.nan), tip_px]),
        "final_depth_um": float(depth_um),
        "theoretical_depth_um": float(p.v0 * p.tau_us),  # m/s * us = um
        "n_pre": n_pre,
    }
    return frames, truth
