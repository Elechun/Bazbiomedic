# 서버 새 채팅 시작 프롬프트

서버에서 저장소를 클론한 뒤 Claude Code 새 채팅에 아래 블록을 그대로 붙여 넣으면 이어서 진행됩니다.

```
너는 이 저장소(Elechun/Bazbiomedic)의 의료 AI 개인 프로젝트를 이어서 진행한다.
먼저 README.md, docs/PLAN.md, docs/agents.md 를 읽고 시작해라.

배경
- 목적: 바즈바이오메딕(무바늘 마이크로젯 CUREJET, 피부·두피 스킨부스터) 전문연구요원(석사, 의료인공지능) 지원용 개인 프로젝트.
- 목표: 회사 미해결 문제 해결이 아니라, 같은 도메인(피부·두피 영상)에서 의료 AI 전 과정을 한 번 끝까지 해 본 유경험자 되기. 난이도는 석사 개인 프로젝트 수준.
- 주제: AI Hub 공개 데이터 '유형별 두피 이미지'로 두피 현미경 사진에서 6개 증상(미세각질·피지과다·모낭사이홍반·모낭홍반농포·비듬·탈모)의 중증도(0~3)를 동시에 예측하고, 시술 전후 변화를 리포트한다.

현재 상태
- scalpai/ 패키지와 scripts/ CLI, tests/ 가 있고 더미 데이터로 8개 테스트와 엔드투엔드(인덱싱→피험자 단위 분할→학습→평가)가 통과한 상태.
- 실제 AI Hub 데이터로는 아직 아무것도 돌리지 않았다. 성능 수치 없음.
- jetlab/ 은 이전 단계의 부록(무바늘 제트 고속영상 합성 실험). 메인과 무관하니 건드리지 마라.

이번 세션에서 할 일 (순서대로, 각 단계 끝나면 짧게 보고)
1. 환경: pip install -r requirements.txt (GPU 면 CUDA torch), python -m pytest -q 로 8 passed 확인.
2. 데이터: data/aihub_scalp/ 에 AI Hub 데이터가 있는지 확인. 없으면 docs/PLAN.md 2절대로 내가 받아야 하니 알려 줘라.
   있으면 JSON 필드명(value_1~6)과 증상 순서, 파일명의 피험자 토큰, 폴더 간 중복을 먼저 확인하고 scalpai/labels.py, prepare_index.py --group-pattern 을 맞춰라.
3. python scripts/prepare_index.py --root data/aihub_scalp --out outputs/index.csv | tee docs/reviews/00_index_report.txt
   누수 0, 중복 제거 수, label_source json 비율, 등급 분포를 보고해라.
4. 예비 학습 2 epoch → 본학습 efficientnet_b0 15 epoch → val 평가 → test 는 마지막에 한 번.
5. docs/agents.md 의 검토 에이전트를 같은 이름·모델로 실행해라: A1(Opus), Astra 주제·방향성(Astra, 없으면 Sonnet), A2(Opus), A3(Opus), A4(Opus), A5a(Astra/Sonnet), A5b(Fable), A6(Opus).
   순서: A1 + Astra 병렬 → A2 → A3 + A4 병렬 → A5a + A5b 병렬 → 확인된 수정 반영·커밋 → A6.
   각 결과는 docs/reviews/<이름>.md 로 저장하고, 마지막에 "에이전트 | 모델 | 담당 | 주요 성과" 표로 정리해라.
6. 커밋은 단계마다, 커밋 메시지에 어느 에이전트가 찾은 버그인지 적어라. 데이터와 outputs/ 는 절대 커밋하지 마라(AI Hub 재배포 금지).

규칙
- 실데이터 없이 성능 수치를 지어내지 마라. 더미 데이터 수치는 "더미" 라고 명시.
- test 셋은 모델 선택에 쓰지 마라.
- 막히면 추측으로 진행하지 말고 무엇이 필요한지 물어라.
```

## 서버 준비 체크리스트

- [ ] `git clone https://github.com/Elechun/Bazbiomedic.git`
- [ ] Python 3.10+ 가상환경, `pip install -r requirements.txt`, GPU 면 CUDA 버전 torch
- [ ] `python -m pytest -q` → 8 passed
- [ ] AI Hub '유형별 두피 이미지' 다운로드 → `data/aihub_scalp/` (https://aihub.or.kr/aidata/30758)
- [ ] Claude Code 새 채팅에 위 프롬프트 붙여넣기
