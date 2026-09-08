"""jetlab: 무바늘 마이크로젯 고속 영상 분석 툴킷.

- synth: 합성 고속 영상 생성 (실제 장비 확보 전 파이프라인 검증용)
- track: 배경 차분 기반 제트 선단 추적, 속도/침투 깊이/제트 폭 추출
- report: 다중 샷 재현성(평균, 표준편차, 변동계수) 리포트
- predict: 분사 조건 -> 침투 깊이 예측 및 목표 깊이 역산
"""

from .synth import JetParams, synthesize_shot
from .track import JetTracker, ShotMetrics

__all__ = ["JetParams", "synthesize_shot", "JetTracker", "ShotMetrics"]
