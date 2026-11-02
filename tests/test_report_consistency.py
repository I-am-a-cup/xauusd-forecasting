"""Проверки полноты аналитического отчёта и актуальности документации."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAME = {"naive": "Naive", "arima": "ARIMA",
        "random_forest": "Random Forest", "mlp": "MLP"}


def test_all_eight_result_rows_match_published_csv():
    doc = (ROOT / "README.md").read_text(encoding="utf-8-sig")
    m = pd.read_csv(ROOT / "results/arima_rf/comparison.csv")
    assert len(m) == 8
    for row in m.itertuples():
        mae = f"{row.mae:.2f}".replace(".", ",")
        rmse = f"{row.rmse:.2f}".replace(".", ",")
        assert f"| {NAME[row.model]} | {row.horizon} | 47 |" in doc
        assert f"| {row.horizon} | 47 |" in doc
        assert mae in doc and rmse in doc


def test_report_figures_exist_and_report_mentions_limits():
    doc = (ROOT / "README.md").read_text(encoding="utf-8-sig")
    for path in [
        "results/report/mae_comparison.png",
        "results/report/temporal_stability.png",
        "results/arima_rf/comparison.png",
        "results/benchmark/model_timings.csv",
    ]:
        assert (ROOT / path).is_file()
        assert path in doc
    for section in ["## Результаты", "## Устойчивость",
                    "## Сопоставимые вычислительные затраты",
                    "## Проверка гипотез", "## Ограничения"]:
        assert section in doc
    assert "праздничн" in doc.lower()


def test_old_todos_no_longer_in_docs():
    for filename in ["README.md", "EXPERIMENT.md", "DATA.md", "EXPERIMENT_PROTOCOL.md"]:
        content = (ROOT / filename).read_text(encoding="utf-8-sig")
        assert "Пока реальные прогнозы не рассчитаны" not in content
        assert "ARIMA, RF или MLP уже реализованы" not in content
        assert "собрать дневной ряд, по которому будем" not in content
