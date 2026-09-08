# 무바늘 마이크로젯 고속 영상 분석 (jetlab)

바즈바이오메딕 전문연구요원 지원용 개인 프로젝트 뼈대입니다.
무바늘 마이크로젯 인젝터의 고속 카메라 영상에서 **제트 속도, 침투 깊이, 제트 폭**을 자동으로 추출하고,
샷 간 **재현성**을 정량화하며, 분사 조건으로부터 **침투 깊이를 예측/역산**하는 파이프라인입니다.

## 왜 이 주제인가

- 회사의 핵심 기술은 초당 수백 m 의 미세 액체 제트로 피부를 뚫는 무바늘 약물전달입니다.
- 관련 논문에서 반복해서 등장하는 과제는 (1) 침투 깊이·용량의 재현성, (2) 실시간 분사 모니터링, (3) 튐/통증 억제입니다.
- 의약품(인슐린, 백신)으로 확장하려면 용량 정확도를 규제기관에 증명해야 하고, 그 근거가 되는 것이 바로 이런 정량 분석입니다.
- 의료인공지능 전공의 강점(영상 처리, 예측 모델)을 이 회사의 물리 문제에 직접 연결하는 주제입니다.

## 구성

```
jetlab/
  synth.py    합성 고속 영상 생성기 (등속 비행 -> 모사체 내 지수 감쇠 모델, 정답 라벨 포함)
  track.py    배경 차분 + Otsu + 밴드 마스크로 제트 선단 추적, 표면 자동 검출
  report.py   다중 샷 재현성 리포트 (평균/표준편차/변동계수, CSV + 그래프)
  predict.py  분사 조건 -> 깊이 회귀 모델, 목표 깊이 -> v0 역산
scripts/run_demo.py   엔드투엔드 데모
tests/                추적 정확도 회귀 테스트
```

## 실행

```bash
pip install -r requirements.txt
python -m pytest -q
python scripts/run_demo.py     # outputs/ 에 shots.csv, summary.csv, repeatability.png 생성
```

현재 데모 결과(합성 데이터 기준): 표면 검출 오차 0 px, 침투 깊이 오차 약 20 um, 깊이 예측 모델 CV R² ≈ 0.98.

## 로드맵

1. **실데이터 연결**: 논문 공개 영상 또는 자체 촬영(젤라틴/PDMS 모사체) 영상으로 `track.py` 검증. 실제 영상은 조명 불균일, 스플래시, 기포가 있어 마스크 로직 보강 필요.
2. **제트 폭·분산 개선**: 현재 폭 추정은 픽셀 해상도와 임계값에 민감함(데모에서 120 um 노즐을 200 um 로 과대 추정). 서브픽셀 프로파일 피팅으로 교체.
3. **튐(splash-back) 정량화**: 표면 위로 되돌아오는 밝기 영역의 면적/시간 곡선을 지표로 추가.
4. **학습 기반 분할**: 규칙 기반 마스크를 소형 U-Net 으로 교체하고, 합성 영상으로 사전학습 후 실영상 소량으로 미세조정.
5. **깊이 예측의 물리 정합**: 문헌의 제트 파워 - 침투 깊이 관계(Schramm-Baxter & Mitragotri)를 피처로 넣어 합성 모델 의존도를 낮춤.
6. **리포트 자동화**: 샷 단위 QC 리포트(PDF)를 생성해 "용량 재현성 근거 문서"의 형태로 마무리.

## 참고 문헌

- Han & Yoh, *A laser based reusable microjet injector for transdermal drug delivery*, J. Appl. Phys. 2010
- Jang et al., *Skin pre-ablation and laser assisted microjet injection for deep tissue penetration*, Lasers Surg. Med. 2017
- Ham & Yoh, *A liquid breakdown driven non-invasive microjet injection system*, 2021
- Lee et al., *A novel needle-free microjet drug injector using Er:YAG LASER*, Clin. Anat. 2022
- Schramm-Baxter & Mitragotri, *Needle-free jet injections: dependence of jet penetration and dispersion in the skin on jet power*, J. Control. Release 2004
