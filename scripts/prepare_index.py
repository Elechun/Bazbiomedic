"""데이터 인덱싱 + 피험자 단위 분할 + 누수/분포 점검.

python scripts/prepare_index.py --root <AI Hub 데이터 루트> --out outputs/index.csv
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scalpai.data import DEFAULT_GROUP_PATTERN, check_leakage, index_dataset, label_distribution, split_by_group

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--out", default="outputs/index.csv")
p.add_argument("--group-pattern", default=DEFAULT_GROUP_PATTERN, help="파일 stem 에서 피험자 id 를 뽑는 정규식(그룹 1)")
p.add_argument("--hash", action="store_true", help="내용 해시로 중복 제거(느림)")
p.add_argument("--val-frac", type=float, default=0.15)
p.add_argument("--test-frac", type=float, default=0.15)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

df = index_dataset(a.root, a.group_pattern, hash_content=a.hash)
if df.empty:
    sys.exit(f"이미지를 찾지 못함: {a.root}")
df = split_by_group(df, a.val_frac, a.test_frac, a.seed)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)

print(f"images: {len(df)}  (duplicates removed: {df.attrs.get('n_duplicates_removed', 0)})")
print("label_source:", df["label_source"].value_counts().to_dict())
print("groups:", df["group"].nunique(), "| split:", df["split"].value_counts().to_dict())
print("leakage check:", json.dumps(check_leakage(df)))
print("\nlabel distribution (rows=symptom, cols=grade 0..3):")
print(label_distribution(df))
for s in ("train", "val", "test"):
    print(f"\n[{s}]"); print(label_distribution(df[df["split"] == s]))
