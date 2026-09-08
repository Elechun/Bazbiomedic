"""AI Hub 레이아웃을 흉내낸 더미 두피 이미지 생성기 (스모크 테스트 전용).

실제 데이터를 대체하지 않는다. 목적은 (1) 인덱싱/분할/학습/평가 코드가 끝까지 도는지,
(2) 피험자 단위 분할과 중복 제거가 동작하는지 확인하는 것뿐이다.

증상별 시각 단서(절차적):
  미세각질  -> 작은 흰 점 개수     피지과다 -> 노란 반투명 얼룩
  홍반      -> 붉은 패치           농포     -> 흰 중심을 가진 붉은 점
  비듬      -> 큰 흰 조각          탈모     -> 머리카락 선 밀도 감소
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .labels import SEVERITY_NAMES_KO, SYMPTOMS, SYMPTOM_NAMES_KO, VALUE_KEYS


def _draw_scalp(rng: np.random.Generator, sev: dict[str, int], size: int = 128) -> Image.Image:
    base = np.array([222, 190, 170], dtype=np.float32) + rng.normal(0, 6, 3)
    img = np.clip(base[None, None, :] + rng.normal(0, 8, (size, size, 3)), 0, 255).astype(np.uint8)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")

    n_hair = int(60 * (1.0 - 0.28 * sev["hairloss"]))
    for _ in range(n_hair):
        x0, y0 = rng.uniform(0, size, 2)
        ang = rng.uniform(0, np.pi)
        L = rng.uniform(20, 60)
        d.line([(x0, y0), (x0 + L * np.cos(ang), y0 + L * np.sin(ang))], fill=(40, 30, 25, 230), width=1)
    for _ in range(sev["erythema"] * 3):
        x, y, r = rng.uniform(0, size), rng.uniform(0, size), rng.uniform(8, 18)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(220, 70, 70, 90))
    for _ in range(sev["sebum"] * 4):
        x, y, r = rng.uniform(0, size), rng.uniform(0, size), rng.uniform(5, 12)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(240, 220, 120, 110))
    for _ in range(sev["microkeratin"] * 25):
        x, y = rng.uniform(0, size, 2)
        d.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(250, 250, 245, 220))
    for _ in range(sev["dandruff"] * 4):
        x, y, r = rng.uniform(0, size), rng.uniform(0, size), rng.uniform(3, 7)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(245, 245, 240, 230))
    for _ in range(sev["pustule"] * 3):
        x, y = rng.uniform(0, size, 2)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(210, 60, 60, 200))
        d.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(255, 250, 230, 255))
    return im


def make_dummy_dataset(root: str | Path, n_subjects: int = 20, imgs_per_subject: int = 4,
                       size: int = 128, seed: int = 0, duplicate_frac: float = 0.1) -> Path:
    """root/01.원천데이터/[원천]k.증상_g.등급/<subj>_<i>.jpg + root/02.라벨링데이터/[라벨].../<same>.json

    - 피험자별로 6개 증상 등급을 뽑고, 같은 피험자의 이미지들은 등급을 공유(약간의 잡음)한다.
    - duplicate_frac 만큼은 같은 이미지를 다른 증상 폴더에 한 번 더 복사해 중복 제거 로직을 시험한다.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    src, lab = root / "01.원천데이터", root / "02.라벨링데이터"
    for s_idx, s in enumerate(SYMPTOMS, 1):
        for g in range(4):
            name = f"{s_idx}.{SYMPTOM_NAMES_KO[s]}_{g}.{SEVERITY_NAMES_KO[g]}"
            (src / f"[원천]{name}").mkdir(parents=True, exist_ok=True)
            (lab / f"[라벨]{name}").mkdir(parents=True, exist_ok=True)

    for subj in range(n_subjects):
        # 등급 분포를 불균형하게: 0 이 가장 많고 3 이 드물게
        base = {s: int(rng.choice(4, p=[0.45, 0.3, 0.17, 0.08])) for s in SYMPTOMS}
        for i in range(imgs_per_subject):
            sev = {s: int(np.clip(v + rng.choice([-1, 0, 0, 0, 1]), 0, 3)) for s, v in base.items()}
            primary = SYMPTOMS[int(rng.integers(6))]
            im = _draw_scalp(rng, sev, size)
            stem = f"S{subj:04d}_{i:02d}"
            folder = f"{SYMPTOMS.index(primary) + 1}.{SYMPTOM_NAMES_KO[primary]}_{sev[primary]}.{SEVERITY_NAMES_KO[sev[primary]]}"
            im.save(src / f"[원천]{folder}" / f"{stem}.jpg", quality=92)
            meta = {"image_id": stem, "image_file_name": f"{stem}.jpg",
                    **{k: str(sev[s]) for k, s in zip(VALUE_KEYS, SYMPTOMS)}}
            with open(lab / f"[라벨]{folder}" / f"{stem}.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
            if rng.uniform() < duplicate_frac:
                other = SYMPTOMS[(SYMPTOMS.index(primary) + 1) % 6]
                folder2 = f"{SYMPTOMS.index(other) + 1}.{SYMPTOM_NAMES_KO[other]}_{sev[other]}.{SEVERITY_NAMES_KO[sev[other]]}"
                im.save(src / f"[원천]{folder2}" / f"{stem}.jpg", quality=92)
                with open(lab / f"[라벨]{folder2}" / f"{stem}.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False)
    return root
