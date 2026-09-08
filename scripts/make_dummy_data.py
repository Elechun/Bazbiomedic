"""더미 데이터 생성: python scripts/make_dummy_data.py --out data/dummy --subjects 30"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scalpai.synth import make_dummy_dataset

p = argparse.ArgumentParser()
p.add_argument("--out", default="data/dummy")
p.add_argument("--subjects", type=int, default=30)
p.add_argument("--per-subject", type=int, default=4)
p.add_argument("--size", type=int, default=128)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
root = make_dummy_dataset(a.out, a.subjects, a.per_subject, a.size, a.seed)
print("dummy dataset ->", root)
