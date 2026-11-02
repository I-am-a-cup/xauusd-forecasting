"""Контракт интеграции: не подменять выбранную MLP конфигурацией по умолчанию."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from src.models.mlp import MLPConfig, load_selected_configs, predict_mlp
from src.validation import holdout_split


def test_selected_configs_match_published_metadata():
    configs = load_selected_configs()
    assert configs == {
        1: MLPConfig((8,), 10.0, 2000),
        5: MLPConfig((8, 4), 10.0, 2000),
    }


def test_missing_metadata_does_not_silently_select_defaults(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_selected_configs(tmp_path / "not-found.json")


def test_invalid_metadata_is_rejected(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"selected": {"1": {"alpha": 1.0}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="h=1 и h=5"):
        load_selected_configs(path)


def test_predict_mlp_without_configs_reproduces_published_selection():
    root = Path(__file__).resolve().parents[1]
    daily = pd.read_csv(root / "data/processed/xauusd_daily_dukascopy.csv")
    split = holdout_split(len(daily))
    with threadpool_limits(limits=1):
        implicit = predict_mlp(daily, split.forecast_origins, last_known_index=split.last_known_index)
        explicit = predict_mlp(daily, split.forecast_origins,
                               last_known_index=split.last_known_index,
                               configs=load_selected_configs())
    assert len(implicit) == 94
    pd.testing.assert_frame_equal(implicit, explicit)
    assert np.isfinite(implicit["y_pred"]).all()


def test_predict_mlp_rejects_partial_configs():
    root = Path(__file__).resolve().parents[1]
    daily = pd.read_csv(root / "data/processed/xauusd_daily_dukascopy.csv")
    split = holdout_split(len(daily))
    with pytest.raises(ValueError, match="обоих горизонтов"):
        predict_mlp(daily, split.forecast_origins[:1],
                    last_known_index=split.last_known_index,
                    configs={1: MLPConfig((8,), 10.0)})
