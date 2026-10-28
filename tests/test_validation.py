"""Проверки защиты от утечки будущих целевых значений."""
import pytest

from src.validation import (
    TemporalSplit, training_origins, holdout_split, expanding_folds
)


def test_training_examples_have_known_targets():
    cutoff = 100
    for h in (1, 5):
        origins = training_origins(cutoff, h, min_history=10)
        assert origins
        assert min(origins) == 9
        assert max(origins) + h == cutoff
        assert all(o+h <= cutoff for o in origins)


def test_holdout_same_origins_and_all_targets_exist():
    split = holdout_split(260, horizons=(1, 5), test_share=0.20)
    assert split.last_known_index == 207
    assert split.forecast_origins[0] == 208
    assert split.forecast_origins[-1] == 254
    for h in (1, 5):
        assert all(origin+h < 260 for origin in split.forecast_origins)
        assert all(train+h <= split.last_known_index
                   for train in split.eligible_train_origins(h))


def test_expanding_folds_do_not_touch_the_holdout():
    folds = expanding_folds(
        n_observations=260,
        initial_train_observations=100,
        validation_size=25,
        step_size=25,
        end_before=208,
        horizons=(1, 5),
    )
    assert len(folds) >= 3
    assert [fold.last_known_index for fold in folds] == sorted(
        fold.last_known_index for fold in folds)
    for fold in folds:
        assert fold.forecast_origins[0] > fold.last_known_index
        for h in (1, 5):
            assert max(fold.forecast_origins)+h < 208
            assert all(o+h <= fold.last_known_index
                       for o in fold.eligible_train_origins(h))


def test_horizon_five_training_excludes_unavailable_labels():
    fold = TemporalSplit(last_known_index=50, forecast_origins=(51, 52))
    assert 49 not in fold.eligible_train_origins(5)
    assert fold.eligible_train_origins(5)[-1] == 45


def test_rejects_short_validation_window():
    with pytest.raises(ValueError, match="длиннее"):
        expanding_folds(100, 50, validation_size=5, step_size=10)


def test_rejects_missing_test_targets():
    with pytest.raises(ValueError, match="Слишком мало тестовых"):
        holdout_split(100, test_share=0.05, horizons=(1, 5), min_train_observations=20)


def test_rejects_invalid_horizons():
    with pytest.raises(ValueError, match="повторяться"):
        holdout_split(200, horizons=(1, 1))
