# 서버 인수인계 계획 (PLAN)

이 문서는 서버에서 바로 이어받아 작업할 수 있도록 전체 계획을 순서대로 적은 것입니다.
현재까지 완료: 코드 뼈대 + 더미 데이터 스모크 테스트(8개 테스트 통과, CLI 엔드투엔드 확인).
아직 안 한 것: 실제 AI Hub 데이터 학습, 검토 에이전트 실행(docs/agents.md), 결과 문서화.

## 0. 목표 한 줄

바즈바이오메딕(무바늘 마이크로젯, 피부·두피 스킨부스터) 지원용으로, 공개 데이터로 두피 상태 중증도 분류 의료 AI를 끝까지 한 번 완성한다.
회사의 미해결 문제를 푸는 것이 아니라 같은 도메인의 유경험자가 되는 것이 목표.

## 1. 서버 환경

```bash
git clone https://github.com/Elechun/Bazbiomedic.git && cd Bazbiomedic
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # GPU 면 CUDA 버전 torch 로 교체
python -m pytest -q                    # 8 passed 여야 함
```

## 2. 데이터 준비 (사람이 해야 하는 단계)

1. AI Hub 로그인 후 '유형별 두피 이미지' 다운로드: https://aihub.or.kr/aidata/30758
2. 압축 해제 후 `data/aihub_scalp/` 아래에 둔다 (예: `data/aihub_scalp/Training/01.원천데이터/[원천]1.미세각질_0.양호/*.jpg`).
3. **먼저 확인할 것** (배포본마다 다를 수 있음):
   - JSON 라벨 필드명이 `value_1`~`value_6` 인지, 증상 순서가 미세각질→피지과다→모낭사이홍반→모낭홍반농포→비듬→탈모 인지. 다르면 `scalpai/labels.py` 의 `VALUE_KEYS`/`SYMPTOMS` 순서를 맞춘다.
   - 파일명에서 피험자/촬영 세션을 구분할 수 있는 토큰이 있는지. 있으면 `prepare_index.py --group-pattern` 에 정규식을 준다. **없으면 피험자 단위 분할이 불가능하므로 이를 한계로 명시**하고 파일명 단위 분할로 진행한다.
   - 같은 이미지가 여러 증상 폴더에 중복 배치돼 있는지 (`prepare_index.py` 출력의 `duplicates removed`). 있으면 정상 동작, 0 이면 `--hash` 로 내용 해시 중복 제거를 한 번 더 확인.
4. 데이터/산출물은 커밋하지 않는다 (`.gitignore` 에 `data/`, `outputs/` 포함, AI Hub 재배포 금지).

## 3. 실행 순서

```bash
# (1) 인덱싱·분할·누수·분포 점검. 출력 전체를 docs/reviews/00_index_report.txt 에 저장
python scripts/prepare_index.py --root data/aihub_scalp --out outputs/index.csv | tee docs/reviews/00_index_report.txt

# (2) 소규모 예비 학습으로 파이프라인·속도 확인 (index.csv 일부만 써도 됨)
python scripts/train.py --index outputs/index.csv --out outputs/pilot --backbone efficientnet_b0 --epochs 2 --batch-size 64 --num-workers 8

# (3) 베이스라인 본학습
python scripts/train.py --index outputs/index.csv --out outputs/effb0 --backbone efficientnet_b0 --epochs 15 --batch-size 64 --num-workers 8

# (4) 평가 (test 는 마지막에 한 번만)
python scripts/evaluate.py --index outputs/index.csv --checkpoint outputs/effb0/best.pt --split val
python scripts/evaluate.py --index outputs/index.csv --checkpoint outputs/effb0/best.pt --split test
```

## 4. 성공 기준 (초안, A6 에이전트가 첫 결과를 보고 재조정)

| 항목 | 기준 |
|---|---|
| 누수 | `check_leakage` 전 항목 0 |
| 최빈값 베이스라인 대비 | 증상 평균 macro-F1 +0.15 이상 |
| 순서형 일치도 | 증상 평균 QWK 0.6 이상 (문헌의 EfficientNet-B0 단일 증상 정확도 75~82% 수준을 참고) |
| 임상적 허용 오차 | within-1 0.90 이상 |
| 재현성 | 같은 seed 두 번 실행 시 val QWK 차이 0.01 이하 |

## 5. 검토 에이전트 실행 (docs/agents.md)

서버에서 Claude Code 로 `docs/agents.md` 의 프롬프트를 순서대로 실행하고, 각 결과를 `docs/reviews/A1.md` … `A6.md`, `Astra_topic.md` 로 저장한다.
순서: A1 + Astra(주제) 병렬 → A2 → A3 + A4 병렬 → A5a(Astra) + A5b(Fable) 병렬 → 수정 반영 → A6.
확인된 버그는 커밋 메시지에 어느 에이전트 발견인지 적는다 (예: `fix(data): group split leak (found by A3)`).

## 6. 확장 로드맵 (베이스라인 완료 후)

1. 순서형 손실 (CORAL / 누적 링크) 로 QWK 개선 비교
2. Grad-CAM 으로 증상별 주목 영역 시각화 → 두피 현미경에서 임상적으로 말이 되는지 확인
3. 시술 전후 비교 리포트 (`scripts/infer.py --after`) 를 PDF 로 출력
4. 경량화 (ONNX/TensorRT) 와 추론 시간 측정 → 클리닉 태블릿 배포 가정
5. (선택) 얼굴 주름 분할 FFHQ-Wrinkle 로 "피부" 쪽 두 번째 모듈

## 7. 지원서에 쓸 스토리 (초안)

- 문제: 두피 시술의 전후 효과를 객관화할 지표가 없다 → 공개 데이터로 6개 증상 중증도 자동 등급화.
- 의료 AI 다운 디테일: 피험자 단위 분할, 중복 제거, 결측 라벨 마스킹, 순서형 지표(QWK), 최빈값 베이스라인 대비 보고.
- 회사 연결: 무바늘 스킨부스터 두피 시술의 전후 정량 리포트로 확장 가능.
- 한계 정직하게: 공개 데이터는 특정 현미경 장비·한국인 표본, 임상 검증 없음.

## 8. 부록: jetlab

`jetlab/` 은 이전 단계에서 만든 무바늘 제트 고속영상 분석 합성 실험입니다. 선행 검토(A1)에서
"기본 침투 깊이 6000 um 는 문헌의 진피 표적(약 300 um) 대비 과대", "R² 0.98 은 합성 규칙을 되찾는 순환 논증" 이 지적됐고
(docs/reviews/jetlab_A1.md), 수정하지 않은 채 부록으로 남겼습니다. 메인 프로젝트와 무관하게 삭제해도 됩니다.
