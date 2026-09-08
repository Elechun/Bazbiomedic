"""증상·등급 정의와 AI Hub 라벨 파싱.

AI Hub '유형별 두피 이미지' 는 이미지 1장당 JSON 1개가 대응하고, JSON 안에
value_1 ~ value_6 (각 0~3) 으로 6개 증상의 중증도가 들어 있다.
폴더명은 "[원천]1.미세각질_2.중등도" 처럼 대표 증상과 등급을 담는다.

주의: 실제 배포본의 필드명/폴더명은 버전에 따라 다를 수 있다. 이 모듈은
value_1~value_6 을 1차로 읽고, 없으면 폴더명에서 대표 증상 1개만 복원한다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SYMPTOMS = ["microkeratin", "sebum", "erythema", "pustule", "dandruff", "hairloss"]
SYMPTOM_NAMES_KO = {
    "microkeratin": "미세각질",
    "sebum": "피지과다",
    "erythema": "모낭사이홍반",
    "pustule": "모낭홍반농포",
    "dandruff": "비듬",
    "hairloss": "탈모",
}
SEVERITY_NAMES_KO = ["양호", "경증", "중등도", "중증"]
NUM_SEVERITY = 4

# AI Hub JSON 의 value_k 순서 (k=1..6) 가 위 SYMPTOMS 순서와 같다고 가정한다.
VALUE_KEYS = [f"value_{i}" for i in range(1, 7)]

_FOLDER_RE = re.compile(r"(\d)\.(미세각질|피지과다|모낭사이홍반|모낭홍반농포|비듬|탈모)_(\d)\.")
_KO_TO_KEY = {v: k for k, v in SYMPTOM_NAMES_KO.items()}


def parse_label_json(path: str | Path) -> dict[str, int]:
    """JSON 라벨 파일 -> {symptom: severity}. value_k 가 없으면 KeyError."""
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    # 일부 배포본은 최상위가 리스트이거나 "images"/"annotations" 로 감싸져 있다.
    if isinstance(obj, list):
        obj = obj[0]
    for wrapper in ("annotations", "label", "labels"):
        if wrapper in obj and isinstance(obj[wrapper], dict):
            obj = {**obj, **obj[wrapper]}
    out = {}
    for sym, key in zip(SYMPTOMS, VALUE_KEYS):
        v = int(obj[key])
        if not 0 <= v < NUM_SEVERITY:
            raise ValueError(f"{path}: {key}={v} out of range")
        out[sym] = v
    return out


def parse_folder_name(folder: str) -> tuple[str, int] | None:
    """'[원천]1.미세각질_2.중등도' -> ('microkeratin', 2). 매칭 안 되면 None."""
    m = _FOLDER_RE.search(folder)
    if not m:
        return None
    return _KO_TO_KEY[m.group(2)], int(m.group(3))
