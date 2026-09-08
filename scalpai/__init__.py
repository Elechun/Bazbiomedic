"""scalpai: AI Hub '유형별 두피 이미지' 기반 두피 상태 다중 증상 중증도 분류.

- labels : 증상/등급 정의, AI Hub JSON 라벨 및 폴더명 파싱
- data   : 데이터 인덱싱(중복 제거), 피험자 단위 분할, torch Dataset
- model  : 공유 백본 + 증상별 6개 헤드(4등급) 다중 태스크 분류기
- train  : 학습 루프(시드 고정, 클래스 가중 CE, CSV/JSON 로깅, best 체크포인트)
- evaluate: 증상별 정확도, macro-F1, QWK(순서형 일치도), 혼동행렬
- synth  : AI Hub 레이아웃을 흉내낸 더미 데이터 생성기(스모크 테스트용)
- infer  : 단일 이미지 추론 및 시술 전후 비교 리포트
"""

from .labels import SYMPTOMS, SYMPTOM_NAMES_KO, SEVERITY_NAMES_KO, NUM_SEVERITY

__all__ = ["SYMPTOMS", "SYMPTOM_NAMES_KO", "SEVERITY_NAMES_KO", "NUM_SEVERITY"]
