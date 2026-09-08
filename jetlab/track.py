"""배경 차분 기반 제트 선단 추적기."""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ShotMetrics:
    tip_y: np.ndarray                 # 프레임별 선단 행 (nan = 미검출)
    velocity_mps: np.ndarray          # 프레임별 선단 속도 [m/s]
    depth_um: np.ndarray              # 프레임별 침투 깊이 [um]
    surface_y: int
    peak_velocity_mps: float
    final_depth_um: float
    jet_width_um: float
    extra: dict = field(default_factory=dict)


class JetTracker:
    def __init__(self, um_per_px: float, fps: float, n_background: int = 8,
                 band_half_width: int = 24, diff_threshold: int | None = None):
        self.um_per_px = um_per_px
        self.fps = fps
        self.n_background = n_background
        self.band_half_width = band_half_width
        self.diff_threshold = diff_threshold

    @staticmethod
    def detect_surface(background: np.ndarray) -> int:
        """행 평균 밝기의 최대 기울기 위치를 모사체 표면으로 판정."""
        row_mean = cv2.GaussianBlur(background, (1, 9), 0).mean(axis=1)
        grad = np.diff(row_mean)
        return int(np.argmax(grad)) + 1

    def _mask(self, frame: np.ndarray, background: np.ndarray, cx: int) -> np.ndarray:
        diff = cv2.absdiff(frame, background)
        diff = cv2.GaussianBlur(diff, (3, 3), 0)
        if self.diff_threshold is None:
            _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        band = np.zeros_like(mask)
        x0, x1 = max(0, cx - self.band_half_width), min(mask.shape[1], cx + self.band_half_width)
        band[:, x0:x1] = 255
        mask = cv2.bitwise_and(mask, band)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return mask

    def track(self, frames: np.ndarray, nozzle_y: int | None = None) -> ShotMetrics:
        frames = np.asarray(frames)
        n = len(frames)
        background = np.median(frames[: self.n_background], axis=0).astype(np.uint8)
        surface_y = self.detect_surface(background)

        # 제트 축 x 위치: 분사 프레임 차분의 열 방향 합이 최대인 열
        agg = np.zeros(frames.shape[1:], dtype=np.float64)
        for f in frames[self.n_background :]:
            agg += cv2.absdiff(f, background)
        cx = int(np.argmax(agg.sum(axis=0)))

        tip_y = np.full(n, np.nan)
        widths = []
        for i in range(self.n_background, n):
            mask = self._mask(frames[i], background, cx)
            rows = np.where(mask.any(axis=1))[0]
            if rows.size == 0:
                continue
            if nozzle_y is not None:
                rows = rows[rows >= nozzle_y]
                if rows.size == 0:
                    continue
            # 연속 구간 중 노즐에서 이어지는 첫 구간의 끝을 선단으로 판정
            breaks = np.where(np.diff(rows) > 3)[0]
            end = rows[breaks[0]] if breaks.size else rows[-1]
            tip_y[i] = float(end)
            mid = (rows[0] + end) // 2
            if end > rows[0] + 4:
                widths.append(int(mask[mid].astype(bool).sum()))

        vel = np.gradient(np.nan_to_num(tip_y, nan=np.nan)) * self.um_per_px * 1e-6 * self.fps
        vel[np.isnan(tip_y)] = np.nan
        depth = np.clip(tip_y - surface_y, 0, None) * self.um_per_px
        valid = ~np.isnan(tip_y)
        return ShotMetrics(
            tip_y=tip_y,
            velocity_mps=vel,
            depth_um=depth,
            surface_y=surface_y,
            peak_velocity_mps=float(np.nanmax(vel)) if valid.any() else float("nan"),
            final_depth_um=float(np.nanmax(depth)) if valid.any() else float("nan"),
            jet_width_um=float(np.median(widths) * self.um_per_px) if widths else float("nan"),
            extra={"axis_x": cx},
        )
