"""학습: python scripts/train.py --index outputs/index.csv --out outputs/run1 --backbone efficientnet_b0 --epochs 10"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from scalpai.train import TrainConfig, train

p = argparse.ArgumentParser()
p.add_argument("--index", required=True)
p.add_argument("--out", default="outputs/run")
for k, v in TrainConfig().__dict__.items():
    if isinstance(v, bool):
        p.add_argument(f"--{k.replace('_', '-')}", type=lambda x: x.lower() in ("1", "true", "yes"), default=v)
    else:
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
a = p.parse_args()
cfg = TrainConfig(**{k: getattr(a, k) for k in TrainConfig().__dict__})
df = pd.read_csv(a.index)
res = train(df, a.out, cfg)
print(json.dumps(res, indent=2, ensure_ascii=False))
