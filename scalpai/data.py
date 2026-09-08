"""데이터 인덱싱, 피험자 단위 분할, torch Dataset."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import SYMPTOMS, parse_folder_name, parse_label_json

IMG_EXT = {".jpg", ".jpeg", ".png"}

# AI Hub 파일명 예: "0001_A2LEBIJDE00001M.jpg" 처럼 앞 토큰이 촬영 세션/피험자 코드인 경우가 많다.
# 확정된 규칙이 아니므로 group_pattern 을 CLI 에서 바꿀 수 있게 둔다.
DEFAULT_GROUP_PATTERN = r"^([^_]+)_"


def _file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def index_dataset(root: str | Path, group_pattern: str = DEFAULT_GROUP_PATTERN,
                  dedupe: bool = True, hash_content: bool = False) -> pd.DataFrame:
    """root 아래 모든 이미지를 찾아 라벨과 함께 표로 만든다.

    - 같은 이미지가 여러 증상 폴더에 중복 배치돼 있으면 파일명(또는 내용 해시) 기준으로 1장만 남긴다.
    - JSON 라벨은 이미지와 같은 stem 을 가진 .json 을 (a) 같은 폴더, (b) '[원천]'->'[라벨]' 대응 폴더에서 찾는다.
    - JSON 이 없으면 폴더명에서 대표 증상 1개만 복원하고 나머지 증상은 -1(결측) 로 둔다.
    """
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        label_path = _find_label(p)
        labels = {s: -1 for s in SYMPTOMS}
        source = "none"
        if label_path is not None:
            try:
                labels.update(parse_label_json(label_path))
                source = "json"
            except (KeyError, ValueError):
                source = "json_invalid"
        if source != "json":
            parsed = parse_folder_name(p.parent.name)
            if parsed:
                labels[parsed[0]] = parsed[1]
                source = "folder" if source == "none" else source
        m = re.match(group_pattern, p.stem)
        group = m.group(1) if m else p.stem
        rows.append({
            "path": str(p),
            "file_name": p.name,
            "group": group,
            "label_source": source,
            **labels,
            "content_hash": _file_md5(p) if hash_content else None,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if dedupe:
        key = "content_hash" if hash_content else "file_name"
        before = len(df)
        # 라벨이 있는 행을 우선 남긴다.
        df = df.sort_values("label_source", key=lambda s: s.map({"json": 0, "folder": 1, "json_invalid": 2, "none": 3}))
        df = df.drop_duplicates(subset=key, keep="first").sort_values("path").reset_index(drop=True)
        df.attrs["n_duplicates_removed"] = before - len(df)
    return df


def _find_label(img: Path) -> Path | None:
    """이미지에 대응하는 JSON 후보를 순서대로 찾는다.

    (a) 같은 폴더의 같은 stem .json
    (b) 상위 디렉터리 '원천데이터'/'01.원천데이터' -> '라벨링데이터'/'02.라벨링데이터' 로 바꾼 경로
    (c) 폴더명 '[원천]' -> '[라벨]' 로 바꾼 경로,  (b)+(c) 동시 적용
    """
    stem_json = img.stem + ".json"
    dir_maps = ((None, None), ("원천데이터", "라벨링데이터"), ("01.원천데이터", "02.라벨링데이터"))
    folder_maps = ((None, None), ("[원천]", "[라벨]"))
    cand = []
    for da, db in dir_maps:
        for fa, fb in folder_maps:
            parts = list(img.parent.parts)
            if da is not None:
                if da not in parts:
                    continue
                parts = [db if x == da else x for x in parts]
            if fa is not None:
                if fa not in parts[-1]:
                    continue
                parts[-1] = parts[-1].replace(fa, fb)
            cand.append(Path(*parts) / stem_json)
    for c in cand:
        if c.exists():
            return c
    return None


def split_by_group(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15,
                   seed: int = 0) -> pd.DataFrame:
    """피험자(group) 단위로 train/val/test 를 나눈다. 같은 피험자의 이미지는 한 split 에만 들어간다."""
    rng = np.random.default_rng(seed)
    groups = np.array(df["group"].unique().tolist(), dtype=object)
    rng.shuffle(groups)
    n = len(groups)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_g = set(groups[:n_test])
    val_g = set(groups[n_test:n_test + n_val])
    split = np.where(df["group"].isin(test_g), "test",
                     np.where(df["group"].isin(val_g), "val", "train"))
    out = df.copy()
    out["split"] = split
    return out


def check_leakage(df: pd.DataFrame) -> dict:
    """split 간 group / file_name 겹침을 세어 반환. 0 이어야 정상."""
    res = {}
    for key in ("group", "file_name"):
        sets = {s: set(df.loc[df["split"] == s, key]) for s in ("train", "val", "test")}
        res[f"{key}_train_val"] = len(sets["train"] & sets["val"])
        res[f"{key}_train_test"] = len(sets["train"] & sets["test"])
        res[f"{key}_val_test"] = len(sets["val"] & sets["test"])
    return res


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """증상별 등급 분포 (결측 -1 제외). 행: 증상, 열: 등급 0~3."""
    tab = {}
    for s in SYMPTOMS:
        v = df[s]
        tab[s] = v[v >= 0].value_counts().reindex(range(4), fill_value=0)
    return pd.DataFrame(tab).T


def class_weights(df: pd.DataFrame, power: float = 0.5) -> dict[str, np.ndarray]:
    """증상별 역빈도 가중치 (power 로 완화). 결측(-1) 은 제외."""
    w = {}
    for s in SYMPTOMS:
        cnt = label_distribution(df).loc[s].values.astype(float) + 1.0
        inv = (cnt.sum() / cnt) ** power
        w[s] = inv / inv.mean()
    return w


# ---------------------------------------------------------------------------
# torch Dataset (torch 는 여기서만 import 해서 인덱싱 유틸은 torch 없이도 쓰게 한다)
# ---------------------------------------------------------------------------

def build_transforms(image_size: int = 224, train: bool = True):
    import torchvision.transforms as T

    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),   # 두피 현미경 영상은 상하 방향 의미가 없다
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([T.Resize(int(image_size * 1.14)), T.CenterCrop(image_size), T.ToTensor(), norm])


class ScalpDataset:
    """DataFrame 행 -> (image_tensor, label_tensor[6], index). 결측 라벨은 -1."""

    def __init__(self, df: pd.DataFrame, transform):
        from torch.utils.data import Dataset  # noqa: F401  (타입 힌트용)
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        import torch
        from PIL import Image

        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        x = self.transform(img)
        y = torch.tensor([int(row[s]) for s in SYMPTOMS], dtype=torch.long)
        return x, y, i
