from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import scipy.stats as stats
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.utils.helpers import get_logger

logger = get_logger(__name__)

DRIFT_FEATURE_COLUMNS = [
    "n_interactions",
    "n_unique_songs",
    "avg_play_count",
    "median_play_count",
    "std_play_count",
    "max_play_count",
    "total_play_count",
    "diversity_ratio",
]


def calculate_data_drift(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    drift_threshold: float = 0.3,
) -> dict:
    """
    Data drift через Evidently.
    reference_features — агрегированные фичи янв+фев (пользователи как строки).
    current_features  — агрегированные фичи одного дня марта.
    """
    cols = [
        c
        for c in DRIFT_FEATURE_COLUMNS
        if c in reference_features.columns and c in current_features.columns
    ]

    if not cols:
        logger.warning("No common columns for drift detection")
        return {
            "is_drift_detected": False,
            "drift_score": 0.0,
            "drift_type": "data_drift",
            "error": "no_common_columns",
        }

    try:
        definition = DataDefinition(numerical_columns=cols)
        ref_ds = Dataset.from_pandas(
            reference_features[cols].dropna(),
            data_definition=definition,
        )
        cur_ds = Dataset.from_pandas(
            current_features[cols].dropna(),
            data_definition=definition,
        )

        report = Report(metrics=[DataDriftPreset()])
        result = report.run(reference_data=ref_ds, current_data=cur_ds)
        result_dict = result.dict()

        drift_share = 0.0
        dataset_drift = False
        per_column = {}

        for metric in result_dict.get("metrics", []):
            metric_id = str(metric.get("metric", ""))
            res = metric.get("result", {})

            if "DatasetDriftMetric" in metric_id:
                dataset_drift = res.get("dataset_drift", False)
                drift_share = float(res.get("drift_share", 0.0))

            if "ColumnDriftMetric" in metric_id:
                col_name = res.get("column_name", "unknown")
                per_column[col_name] = {
                    "drift_detected": res.get("drift_detected", False),
                    "stattest": res.get("stattest_name", ""),
                    "p_value": res.get("p_value", None),
                }

        return {
            "is_drift_detected": dataset_drift,
            "drift_score": drift_share,
            "drift_type": "data_drift",
            "n_features_drifted": sum(
                1 for v in per_column.values() if v["drift_detected"]
            ),
            "n_features_total": len(cols),
            "per_column": per_column,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Data drift calculation error: {e}")
        return {
            "is_drift_detected": False,
            "drift_score": 0.0,
            "drift_type": "data_drift",
            "error": str(e),
        }


def calculate_concept_drift(
    current_metrics: dict,
    baseline_metrics: dict,
    threshold_pct: float = 0.1,
) -> dict:
    """
    Concept drift как деградация метрик.
    current_metrics  — метрики модели на данных текущего дня.
    baseline_metrics — эталонные метрики (конец февраля / среднее за период).
    threshold_pct    — порог падения, 0.1 = 10%.

    Сравниваем только метрики с @k суффиксом.
    """
    key_metrics = [
        "ndcg@10",
        "hit_rate@10",
        "mrr@10",
        "precision@10",
        "recall@10",
        "map@10",
    ]

    degraded = {}
    for key in key_metrics:
        baseline_val = baseline_metrics.get(key)
        current_val = current_metrics.get(key)

        if baseline_val is None or current_val is None:
            continue
        if baseline_val == 0:
            continue

        drop_pct = (baseline_val - current_val) / baseline_val
        if drop_pct > threshold_pct:
            degraded[key] = {
                "baseline": round(float(baseline_val), 4),
                "current": round(float(current_val), 4),
                "drop_pct": round(drop_pct * 100, 2),
            }

    n_degraded = len(degraded)
    drift_score = n_degraded / len(key_metrics) if key_metrics else 0.0

    return {
        "is_drift_detected": n_degraded > 0,
        "drift_score": drift_score,
        "drift_type": "concept_drift",
        "n_metrics_degraded": n_degraded,
        "degraded_metrics": degraded,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def detect_statistical_anomaly(
    current_value: float,
    historical_values: list[float],
    confidence: float = 0.95,
) -> dict:
    """
    Проверяет является ли current_value аномальным относительно
    исторических значений через доверительный интервал.

    historical_values — список значений за прошлые дни (например, янв+фев).
    Возвращает словарь с результатом проверки.
    """
    if len(historical_values) < 7:
        return {
            "is_anomaly": False,
            "reason": "not_enough_history",
            "lower_bound": None,
            "upper_bound": None,
        }

    arr = np.array(historical_values, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1)

    # t-интервал для малых выборок
    t_crit = stats.t.ppf((1 + confidence) / 2, df=len(arr) - 1)
    lower = mean - t_crit * std
    upper = mean + t_crit * std

    is_anomaly = float(current_value) < lower or float(current_value) > upper

    return {
        "is_anomaly": is_anomaly,
        "current_value": float(current_value),
        "mean": round(float(mean), 4),
        "std": round(float(std), 4),
        "lower_bound": round(float(lower), 4),
        "upper_bound": round(float(upper), 4),
        "confidence": confidence,
    }


def get_retraining_window(
    current_date: str,
    march_dir: str,
    january_path: str,
    february_path: str,
    window_days: int = 45,
) -> tuple[pl.DataFrame, str, str]:
    """
    Собирает rolling window для переобучения:
    "последние window_days дней до current_date".

    Возвращает (DataFrame с данными, дата начала, дата конца).
    """
    from datetime import datetime, timedelta

    end_date = datetime.strptime(current_date, "%Y-%m-%d")
    start_date = end_date - timedelta(days=window_days)

    frames = []

    # Январь
    jan_df = pl.read_parquet(january_path).select(["user_id", "song_id", "ts"])
    jan_df = jan_df.filter(
        (pl.col("ts") >= pl.lit(start_date)) & (pl.col("ts") < pl.lit(end_date))
    )
    if jan_df.height > 0:
        frames.append(jan_df)

    # Февраль
    feb_df = pl.read_parquet(february_path).select(["user_id", "song_id", "ts"])
    feb_df = feb_df.filter(
        (pl.col("ts") >= pl.lit(start_date)) & (pl.col("ts") < pl.lit(end_date))
    )
    if feb_df.height > 0:
        frames.append(feb_df)

    # Мартовские дни до current_date
    march_path = Path(march_dir)
    for fname in sorted(march_path.glob("march_*.parquet")):
        day_df = pl.read_parquet(str(fname)).select(["user_id", "song_id", "ts"])
        day_df = day_df.filter(
            (pl.col("ts") >= pl.lit(start_date)) & (pl.col("ts") < pl.lit(end_date))
        )
        if day_df.height > 0:
            frames.append(day_df)

    if not frames:
        raise ValueError(f"No data found for window {start_date} — {end_date}")

    combined = pl.concat(frames)
    return combined, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
