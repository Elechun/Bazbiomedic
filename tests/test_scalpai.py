import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scalpai.data import check_leakage, index_dataset, label_distribution, split_by_group
from scalpai.evaluate import per_symptom_metrics
from scalpai.labels import SYMPTOMS, parse_folder_name, parse_label_json
from scalpai.synth import make_dummy_dataset


@pytest.fixture(scope="module")
def dummy(tmp_path_factory):
    root = tmp_path_factory.mktemp("d") / "dummy"
    make_dummy_dataset(root, n_subjects=12, imgs_per_subject=3, size=64, seed=1, duplicate_frac=0.5)
    return root


def test_parse_folder_name():
    assert parse_folder_name("[원천]3.모낭사이홍반_2.중등도") == ("erythema", 2)
    assert parse_folder_name("random") is None


def test_parse_label_json(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"value_1": "0", "value_2": 1, "value_3": "3", "value_4": 2, "value_5": 0, "value_6": "1"}))
    lab = parse_label_json(p)
    assert [lab[s] for s in SYMPTOMS] == [0, 1, 3, 2, 0, 1]


def test_index_dedupes_and_reads_json(dummy):
    df = index_dataset(dummy)
    assert df.attrs["n_duplicates_removed"] > 0
    assert df["file_name"].is_unique
    assert (df["label_source"] == "json").all()
    assert (df[SYMPTOMS] >= 0).all().all()
    assert len(df) == 12 * 3


def test_group_split_no_leakage(dummy):
    df = split_by_group(index_dataset(dummy), 0.2, 0.2, seed=3)
    leak = check_leakage(df)
    assert all(v == 0 for v in leak.values()), leak
    assert set(df["split"]) == {"train", "val", "test"}


def test_metrics_perfect_and_missing():
    yt = np.array([[0, 1, 2, 3, -1, 0], [1, 1, 2, 3, -1, 3]])
    m = per_symptom_metrics(yt, yt.copy())
    assert m.loc["microkeratin", "acc"] == 1.0
    assert m.loc["dandruff", "n"] == 0


def test_smoke_train_tiny(dummy, tmp_path):
    import torch
    from scalpai.train import TrainConfig, evaluate_split, train

    df = split_by_group(index_dataset(dummy), 0.25, 0.25, seed=0)
    cfg = TrainConfig(backbone="tiny", pretrained=False, image_size=64, batch_size=8, epochs=2,
                      num_workers=0, device="cpu", patience=5)
    res = train(df, tmp_path / "run", cfg)
    assert Path(res["checkpoint"]).exists()
    log = pd.read_csv(res["log"])
    assert len(log) == 2 and np.isfinite(log["train_loss"]).all()
    met, yt, yp = evaluate_split(df, res["checkpoint"], "test", num_workers=0)
    assert yt.shape == yp.shape and yp.min() >= 0 and yp.max() <= 3
