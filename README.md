# ScalpAI — 두피 상태 다중 증상 중증도 분류 (바즈바이오메딕 지원용 개인 프로젝트)

AI Hub 공개 데이터 **'유형별 두피 이미지'** 로 두피 현미경 사진 한 장에서 6개 증상
(미세각질·피지과다·모낭사이홍반·모낭홍반농포·비듬·탈모)의 중증도(양호/경증/중등도/중증)를 동시에 예측하고,
시술 전후 사진을 비교해 변화량을 리포트하는 의료 AI 프로젝트입니다.

> 서버에서 이어서 작업하려면 **[docs/PLAN.md](docs/PLAN.md)** 부터 읽으세요.
> 검토 에이전트 구성과 프롬프트는 **[docs/agents.md](docs/agents.md)** 에 있습니다.

## 왜 이 주제인가

- 바즈바이오메딕의 무바늘 마이크로젯(CUREJET)은 스킨부스터를 피부·두피에 전달하는 장비이고, 회사가 공개적으로 밝힌 확장 분야에 **두피 치료**가 있습니다.
- 두피 시술의 앞뒤에는 "지금 두피 상태가 어떤가"와 "시술 후 얼마나 좋아졌나"를 객관화하는 문제가 붙습니다. 이 프로젝트는 그 문제를 공개 데이터로 경험하는 것이 목적입니다.
- 회사가 해결 못 한 문제를 푸는 것이 아니라, **같은 도메인(피부·두피 영상)에서 의료 AI 전 과정을 한 번 끝까지 해 본 사람**이 되는 것이 목표입니다.
- 난이도: 전이학습 기반 이미지 분류라 석사 개인 프로젝트로 무리가 없고, 순서형 평가 지표(QWK)와 피험자 단위 분할 같은 "의료 AI 다운 디테일"을 보여줄 수 있습니다.

## 구성

```
scalpai/
  labels.py    증상·등급 정의, AI Hub JSON(value_1~6)/폴더명 파싱
  data.py      인덱싱(중복 제거), 피험자 단위 split, 누수 점검, torch Dataset/transform
  model.py     EfficientNet-B0 / ResNet18 / tiny 백본 + 증상별 6 헤드, 결측 라벨 마스킹 손실
  train.py     시드 고정, 클래스 가중 CE, CSV 로그, best 체크포인트(평균 QWK), 조기 종료
  evaluate.py  증상별 acc / within-1 / macro-F1 / QWK, 혼동행렬, 최빈값 베이스라인
  infer.py     단일 이미지 추론, 시술 전후 비교 리포트
  synth.py     AI Hub 레이아웃을 흉내낸 더미 데이터(스모크 테스트 전용)
scripts/       make_dummy_data / prepare_index / train / evaluate / infer CLI
tests/         단위 + 스모크 테스트 (더미 데이터로 학습까지 실행)
docs/          PLAN.md(서버 인수인계), agents.md(검토 에이전트), reviews/(검토 결과)
jetlab/        부록: 무바늘 제트 고속영상 분석 합성 실험 (별도 README)
```

## 빠른 실행 (더미 데이터, CPU 1분 내)

```bash
pip install -r requirements.txt
python -m pytest -q
python scripts/make_dummy_data.py --out data/dummy --subjects 30
python scripts/prepare_index.py --root data/dummy --out outputs/dummy/index.csv
python scripts/train.py --index outputs/dummy/index.csv --out outputs/dummy/run --backbone tiny --pretrained false --image-size 96 --epochs 3 --num-workers 0
python scripts/evaluate.py --index outputs/dummy/index.csv --checkpoint outputs/dummy/run/best.pt --split test
```

## 실제 데이터 실행 (서버)

```bash
python scripts/prepare_index.py --root data/aihub_scalp --out outputs/index.csv          # 인덱싱·분할·누수 점검
python scripts/train.py --index outputs/index.csv --out outputs/effb0 --backbone efficientnet_b0 --epochs 15
python scripts/evaluate.py --index outputs/index.csv --checkpoint outputs/effb0/best.pt --split test
python scripts/infer.py --checkpoint outputs/effb0/best.pt --image before.jpg --after after.jpg
```

## 데이터

- AI Hub '유형별 두피 이미지' (https://aihub.or.kr/aidata/30758): 두피 현미경 이미지 약 10만 장, 이미지당 JSON 라벨 1개(6개 증상 × 0~3 등급).
- AI Hub 데이터는 **재배포 금지**입니다. 이 저장소에는 데이터와 학습 산출물을 커밋하지 않습니다(.gitignore).
- 현재 상태: 더미 데이터로 파이프라인만 검증했고, 실제 데이터 성능 수치는 아직 없습니다.

## 지표

| 지표 | 의미 |
|---|---|
| QWK | 순서형 등급 일치도. 두피·피부 등급 논문의 표준 지표. 모델 선택 기준 |
| within-1 | 한 등급 이내 오차 비율. 임상적으로 "크게 틀리지 않음" |
| macro-F1 | 소수 등급(중증)을 무시하지 못하게 하는 지표 |
| 최빈값 베이스라인 | 항상 최빈 등급을 찍는 모델. 이보다 못하면 학습이 안 된 것 |
