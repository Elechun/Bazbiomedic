"""평가: python scripts/evaluate.py --index outputs/index.csv --checkpoint outputs/run1/best.pt --split test"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scalpai.evaluate import confusion_matrices, majority_baseline
from scalpai.labels import SYMPTOMS
from scalpai.train import evaluate_split

p = argparse.ArgumentParser()
p.add_argument("--index", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--split", default="test")
p.add_argument("--out", default=None)
a = p.parse_args()

df = pd.read_csv(a.index)
met, yt, yp = evaluate_split(df, a.checkpoint, a.split)
y_train = df.loc[df["split"] == "train", SYMPTOMS].values
base = majority_baseline(y_train, yt)
print(f"== {a.split} metrics (model) =="); print(met.round(3))
print(f"\n== {a.split} metrics (majority baseline) =="); print(base.round(3))
print("\nmean over symptoms: model qwk %.3f / f1 %.3f | baseline qwk %.3f / f1 %.3f"
      % (met["qwk"].mean(), met["macro_f1"].mean(), base["qwk"].mean(), base["macro_f1"].mean()))
out = Path(a.out or Path(a.checkpoint).parent)
met.to_csv(out / f"{a.split}_metrics.csv"); base.to_csv(out / f"{a.split}_baseline.csv")
cms = {s: cm.tolist() for s, cm in confusion_matrices(yt, yp).items()}
with open(out / f"{a.split}_confusion.json", "w") as f:
    json.dump(cms, f, indent=1)
print("saved ->", out)
