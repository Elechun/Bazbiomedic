"""추론: python scripts/infer.py --checkpoint outputs/run1/best.pt --image a.jpg [--after b.jpg]"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scalpai.infer import before_after_report, load_model, predict_image

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--image", required=True)
p.add_argument("--after", default=None, help="시술 후 이미지 (있으면 전후 비교 리포트)")
a = p.parse_args()
model, cfg = load_model(a.checkpoint)
if a.after:
    print(before_after_report(model, cfg, a.image, a.after))
else:
    print(json.dumps(predict_image(model, cfg, a.image), indent=1, ensure_ascii=False))
