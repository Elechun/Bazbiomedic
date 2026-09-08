# 검토 에이전트 구성

이전 프로젝트(뷰웍스, NIH 흉부 X-ray)에서 쓴 구성을 그대로 가져오고, 데이터셋 주제만 바즈바이오메딕(두피 영상)으로 바꾼 것입니다.
Astra 모델이 없는 환경에서는 Astra 역할을 Sonnet 으로 대체합니다.

| 에이전트 | 모델 | 담당 | 실행 시점 |
|---|---|---|---|
| A1 주제·포지셔닝 | Opus | 바즈바이오메딕 공고 확인, README 평가 | 처음 |
| Astra 주제·방향성 교차검증 | Astra (없으면 Sonnet) | A1 과 독립적으로 주제·데이터셋 선택의 타당성, 대안 비교 | A1 과 병렬 |
| A2 데이터 적합성 | Opus | AI Hub 두피 데이터 특성, 분포, 편향, 대안 데이터셋 | 인덱싱 후 |
| A3 전처리·통계 | Opus | RNG, split, 중복, 표시 변환 | 예비 학습 후 |
| A4 모델·최적화 | Opus | 학습 설정, 속도, 병목 | 예비 학습 후 |
| A5a 방법론 교차검증 | Astra (없으면 Sonnet) | 수치·알고리즘 검산 | 본학습 후 |
| A5b 방법론 교차검증 | Fable | 수치·알고리즘 검산 (독립) | A5a 와 병렬 |
| A6 목표·진단 설계 | Opus | 성공 기준, 로깅, 실패 진단 | 마지막 |

공통 규칙: 읽기 전용(파일 수정 금지), 근거는 file:line 과 실행 수치로, 확신 없는 지적은 "추정" 표시, 한국어 500단어 이내. 결과는 `docs/reviews/<이름>.md` 로 저장.

---

## A1 주제·포지셔닝 (Opus)

```
역할: A1 주제·포지셔닝 검토 (읽기 전용).
맥락: 이 저장소는 바즈바이오메딕(무바늘 마이크로젯 CUREJET, 피부·두피 스킨부스터, 인슐린/백신/두피치료 확장 계획) 전문연구요원(석사, 의료인공지능 전공) 지원용 개인 프로젝트다. README.md, docs/PLAN.md 를 읽어라.
할 일:
1. 웹 검색으로 바즈바이오메딕 최근 채용 공고(연구원/전문연구요원)의 담당업무·자격요건·우대사항을 확인하고, 확인 못 한 부분은 명시.
2. README 의 주제 선정 근거와 회사 방향의 정합성 평가. 근거 없는 주장, 잘못된 인용 지적.
3. 지원서에서 이 프로젝트가 어떻게 읽힐지, 가장 큰 약점 1~2개와 개선안.
출력: (a) 공고 확인 사실 vs 미확인, (b) README 검증, (c) 약점·개선안, 출처 URL.
```

## Astra 주제·방향성 교차검증 (Astra / Sonnet)

```
역할: Astra, 주제·방향성 교차검증 (읽기 전용, A1 결과를 보지 않은 상태로 독립 판단).
맥락: 위 A1 과 동일. 추가로 목표는 "회사 미해결 문제 해결" 이 아니라 "같은 도메인 유경험자 되기", 난이도는 석사 개인 프로젝트 수준.
할 일:
1. 후보 주제 3개 이상을 표로 비교: (a) AI Hub 두피 이미지 중증도 분류(현재 선택), (b) FFHQ-Wrinkle 얼굴 주름 분할, (c) 피부 병변 분류(HAM10000 등), (d) 인슐린/혈당 예측(simglucose 등), 그 외. 기준: 회사 주제 연관성, 공개 데이터 접근성·라이선스, 난이도, 지원서 설득력, 흔함(차별성).
2. 현재 선택이 최선인지 판정하고, 아니라면 바꿔야 할 이유와 대안을 제시. 최선이면 보강할 점 2개.
3. docs/PLAN.md 의 성공 기준(4절)이 현실적인지 문헌 수치로 검증.
출력: 비교표 + 판정 + 근거 URL. 500단어 이내.
```

## A2 데이터 적합성 (Opus)

```
역할: A2 데이터 적합성 검토 (읽기 전용, scratch 디렉터리에서 실험 허용).
대상: data/aihub_scalp 실데이터와 docs/reviews/00_index_report.txt, scalpai/data.py, scalpai/labels.py.
할 일:
1. 데이터 특성: 이미지 해상도·장비·촬영 조건, 증상별 등급 분포와 불균형 정도, 증상 간 상관(예: 홍반과 농포 동시 발생), 결측 라벨 비율.
2. 편향: 같은 피험자/세션 이미지가 여러 폴더에 중복되는지, 폴더명 대표 증상과 JSON value 가 불일치하는 사례 수.
3. 피험자 단위 분할이 실제로 가능한지(파일명 규칙 확인). 불가능하면 대안(세션 단위, 촬영일 단위) 제시.
4. 대안·보강 데이터셋 조사 (국내외 두피/피부 공개 데이터) 와 각각의 접근성·라이선스.
출력: 수치 표 + file:line 근거 + "가장 중요한 발견 1개".
```

## A3 전처리·통계 (Opus)

```
역할: A3 전처리·통계 검토 (읽기 전용, scratch 실험 허용).
대상: scalpai/data.py, scalpai/train.py, scalpai/evaluate.py, outputs/pilot/*.
점검: (1) 시드/RNG: set_seed, DataLoader generator, worker_init_fn 이 재현성을 보장하는지 (같은 seed 두 번 실행해 val 지표 차이 측정). (2) split: group 분할 후 check_leakage 결과, 증상별 등급 분포가 split 간 비슷한지(층화 필요 여부). (3) 표시·변환: Normalize 통계(ImageNet)가 두피 현미경 영상에 적절한지, RandomVerticalFlip/ColorJitter 가 라벨 의미(홍반 색)를 훼손하지 않는지. (4) 지표: QWK 계산에서 단일 클래스일 때 NaN 처리, class_weights 의 +1 스무딩과 power 가 합리적인지.
각 항목 "버그/오해/문제없음" 판정 + 근거 + 최소 diff.
```

## A4 모델·최적화 (Opus)

```
역할: A4 모델·최적화 검토 (읽기 전용, scratch 실험 허용).
대상: scalpai/model.py, scalpai/train.py, outputs/pilot/train_log.csv.
점검: (1) 학습 설정: lr 3e-4/AdamW/cosine/epochs 가 EfficientNet-B0 미세조정에 적절한지, backbone 과 head 의 lr 분리 필요 여부. (2) 손실: 6 헤드 평균 CE 가 특정 증상에 지배되는지(loss_* 열로 확인), 순서형 손실 대안. (3) 속도: epoch 당 시간, DataLoader 병목(num_workers, 이미지 디코딩, Resize 크기), AMP 적용 여부. 2배 이상 빨라질 최소 수정안. (4) grad_clip, dropout, 클래스 가중치 power 의 실제 효과를 소규모 ablation 으로 수치화.
출력: 실험 표 + 결론 + diff + "진짜 원인" 한 문장.
```

## A5a 방법론 교차검증 (Astra / Sonnet)

```
역할: A5a 방법론 교차검증, 독립 검토자 (읽기 전용, 다른 검토 결과를 보지 않음).
대상: scalpai/*, docs/PLAN.md, outputs/effb0/*.
할 일: 수치·알고리즘 정확성을 의심하며 검산. (1) per_symptom_metrics 의 QWK/within-1/macro-F1 정의와 결측 처리 검산. (2) multitask_loss 의 결측 마스킹과 헤드 수 나눗셈이 배치마다 달라질 때 gradient 스케일 문제. (3) split_by_group 의 비율 반올림과 소수 그룹 처리. (4) README/PLAN 의 수치 주장이 outputs 로그와 일치하는지. (5) tests/ 허용오차.
심각도 순 번호, 근거(file:line, 검산식/실행 수치), 수정안, 확신 없으면 "추정".
```

## A5b 방법론 교차검증 (Fable)

```
(A5a 와 동일한 프롬프트. 독립적으로 실행하고 결과를 A5a 와 대조해 일치/불일치 항목을 표로 정리한다.)
```

## A6 목표·진단 설계 (Opus)

```
역할: A6 목표·진단 설계.
입력: docs/reviews/A1~A5b, outputs/effb0/{train_log.csv,val_metrics_best.csv,test_metrics.csv,test_baseline.csv}.
할 일: (1) docs/PLAN.md 4절 성공 기준을 실제 결과로 재조정(너무 쉬움/불가능 판정). (2) 학습 실패를 조기에 잡을 진단 규칙 설계: train/val 지표 동일 여부(누수 신호), 헤드별 손실 정체, 예측이 최빈값으로 붕괴(collapse) 감지, seed 간 분산. (3) 로깅에 추가할 항목(헤드별 val QWK, 예측 분포 히스토그램, 혼동행렬 저장 주기). (4) 다음 2주 실행 계획.
출력: 수정된 성공 기준 표 + 진단 규칙 + 로깅 diff + 계획.
```
