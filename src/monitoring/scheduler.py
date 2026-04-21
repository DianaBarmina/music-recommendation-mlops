import json
import os
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
import polars as pl
from scipy.sparse import load_npz

from src.monitoring.daily_evaluator import evaluate_model_on_day
from src.monitoring.drift_detector import (
    calculate_concept_drift,
    calculate_data_drift,
    detect_statistical_anomaly,
)
from src.monitoring.feature_engineering import (
    aggregate_daily_features,
    aggregate_reference_features,
    compute_dataset_stats,
)
from src.monitoring.report_generator import generate_drift_report
from src.utils.helpers import get_logger, load_params

logger = get_logger(__name__)


_ALLOWED = re.compile(r"[^0-9a-zA-Z_\-./ ]+")


def _sanitize(name: str) -> str:
    name = name.replace("@", "_at_")
    return _ALLOWED.sub("_", name)


def _sanitize_metrics(d: dict) -> dict:
    return {_sanitize(k): float(v) for k, v in d.items() if isinstance(v, (int, float))}


class DailyMonitor:
    """
    Запускается при добавлении нового дня марта
    Инкапсулирует всю логику мониторинга одного дня
    """

    def __init__(self, db_session, prometheus_metrics: dict):
        self.db = db_session
        self.metrics = prometheus_metrics
        self.params = load_params()

    def _load_model_artifacts(self):
        model_path = self.params["model"]["model_path"]
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        train_matrix = load_npz(self.params["data"]["train_matrix_path"])
        users_map = pl.read_parquet(self.params["data"]["user_mapping_path"])
        items_map = pl.read_parquet(self.params["data"]["item_mapping_path"])
        return model, train_matrix, users_map, items_map

    def _load_reference_features(self) -> "pd.DataFrame":
        return aggregate_reference_features(self.params["data"]["reference_path"])

    def _get_baseline_metrics(self) -> dict:
        metrics_path = Path("metrics.json")
        if not metrics_path.exists():
            return {}
        with open(metrics_path) as f:
            all_metrics = json.load(f)
        return {
            k.replace("als_test_", ""): v
            for k, v in all_metrics.items()
            if k.startswith("als_test_")
        }

    def _get_historical_stats(self, stat_name: str) -> list[float]:
        """
        Возвращает исторические значения stat_name из таблицы daily_stats.
        Используется для расчёта доверительного интервала.
        """
        from services.api.models_db import DailyStats

        rows = self.db.query(DailyStats).order_by(DailyStats.data_date).all()
        values = []
        for row in rows:
            val = getattr(row, stat_name, None)
            if val is not None:
                values.append(float(val))
        return values

    def _save_drift_report(
        self,
        drift_result: dict,
        data_date: str,
        report_path: str = "",
    ):
        from services.api.models_db import DriftReport

        report = DriftReport(
            data_date=data_date,
            drift_type=drift_result.get("drift_type", "unknown"),
            drift_score=drift_result.get("drift_score", 0.0),
            is_drift_detected=drift_result.get("is_drift_detected", False),
            report_path=report_path,
            details=drift_result,
        )
        self.db.add(report)
        self.db.commit()

    def _save_daily_stats(self, data_date: str, stats: dict, metrics: dict):
        from services.api.models_db import DailyStats

        existing = (
            self.db.query(DailyStats).filter(DailyStats.data_date == data_date).first()
        )
        if existing:
            row = existing
        else:
            row = DailyStats(data_date=data_date)
            self.db.add(row)

        row.n_interactions = stats.get("n_interactions")
        row.n_unique_users = stats.get("n_unique_users")
        row.n_unique_items = stats.get("n_unique_items")
        row.avg_play_count = stats.get("avg_play_count")
        row.median_play_count = stats.get("median_play_count")
        row.std_play_count = stats.get("std_play_count")

        if metrics:
            row.ndcg_at_10 = metrics.get("ndcg@10")
            row.hit_rate_at_10 = metrics.get("hit_rate@10")
            row.mrr_at_10 = metrics.get("mrr@10")
            row.precision_at_10 = metrics.get("precision@10")
            row.recall_at_10 = metrics.get("recall@10")
            row.map_at_10 = metrics.get("map@10")
            row.n_eval_users = metrics.get("n_eval_users")

        self.db.commit()

    def _create_alert(
        self,
        data_date: str,
        alert_type: str,
        severity: str,
        message: str,
        metric_name: str | None = None,
        metric_value: float | None = None,
        threshold: float | None = None,
    ):
        from services.api.models_db import Alert

        alert = Alert(
            data_date=data_date,
            alert_type=alert_type,
            severity=severity,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            message=message,
        )
        self.db.add(alert)
        self.db.commit()

        if "ALERTS_TOTAL" in self.metrics:
            self.metrics["ALERTS_TOTAL"].labels(
                alert_type=alert_type,
                severity=severity,
            ).inc()

        logger.warning(f"ALERT [{severity}] {alert_type}: {message}")

    def _log_to_mlflow(
        self,
        data_date: str,
        stats: dict,
        daily_metrics: dict,
        drift_results: list[dict],
    ):
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI", self.params["mlflow"]["tracking_uri"]
        )
        mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(self.params["mlflow"]["experiment_name"] + "-monitoring")

        with mlflow.start_run(run_name=f"monitoring_{data_date}"):
            mlflow.set_tag("data_date", data_date)
            mlflow.set_tag("run_type", "daily_monitoring")

            mlflow.log_metrics(
                _sanitize_metrics({f"data_{k}": v for k, v in stats.items()})
            )

            if daily_metrics:
                mlflow.log_metrics(
                    _sanitize_metrics(
                        {f"daily_{k}": v for k, v in daily_metrics.items()}
                    )
                )

            """
            mlflow.log_metrics(
                {
                    f"data_{k}": float(v)
                    for k, v in stats.items()
                    if isinstance(v, (int, float))
                }
            )

            if daily_metrics:
                mlflow.log_metrics(
                    {
                        f"daily_{k}": float(v)
                        for k, v in daily_metrics.items()
                        if isinstance(v, (int, float))
                    }
                )
            """

            for dr in drift_results:
                prefix = dr.get("drift_type", "drift")
                mlflow.log_metrics(
                    {
                        f"{prefix}_score": dr.get("drift_score", 0.0),
                        f"{prefix}_detected": int(dr.get("is_drift_detected", False)),
                    }
                )

    def _update_prometheus(
        self,
        stats: dict,
        daily_metrics: dict,
        drift_results: list[dict],
        minutes_since_retrain: float,
    ):
        m = self.metrics

        if "DATA_N_INTERACTIONS" in m:
            m["DATA_N_INTERACTIONS"].set(stats.get("n_interactions", 0))
        if "DATA_N_USERS" in m:
            m["DATA_N_USERS"].set(stats.get("n_unique_users", 0))

        for dr in drift_results:
            drift_type = dr.get("drift_type", "unknown")
            if "DRIFT_SCORE" in m:
                m["DRIFT_SCORE"].labels(drift_type=drift_type).set(
                    dr.get("drift_score", 0.0)
                )
            if "DRIFT_DETECTED" in m:
                m["DRIFT_DETECTED"].labels(drift_type=drift_type).set(
                    1 if dr.get("is_drift_detected") else 0
                )

        if daily_metrics and "MODEL_METRIC" in m:
            for metric_name, value in daily_metrics.items():
                if isinstance(value, float):
                    m["MODEL_METRIC"].labels(
                        metric_name=metric_name, split="daily"
                    ).set(value)

        if "MINUTES_SINCE_RETRAIN" in m:
            m["MINUTES_SINCE_RETRAIN"].set(minutes_since_retrain)

    def process_day(self, data_date: str, day_parquet_path: str) -> dict:
        """
        Главный метод — обрабатывает один день марта.
        Возвращает сводку результатов.
        """
        logger.info(f"Processing day: {data_date}")

        day_df = pl.read_parquet(day_parquet_path)
        if "play_count" not in day_df.columns:
            day_df = day_df.with_columns(pl.lit(1).alias("play_count"))

        model, train_matrix, users_map, items_map = self._load_model_artifacts()
        reference_features = self._load_reference_features()

        stats = compute_dataset_stats(day_df)
        current_features = aggregate_daily_features(day_df)

        anomaly_details = {}
        for stat_name in ["n_interactions", "n_unique_users", "avg_play_count"]:
            historical = self._get_historical_stats(stat_name)
            current_val = stats.get(stat_name, 0)
            anomaly_result = detect_statistical_anomaly(current_val, historical)
            if anomaly_result["is_anomaly"]:
                anomaly_details[stat_name] = anomaly_result
                self._create_alert(
                    data_date=data_date,
                    alert_type="data_anomaly",
                    severity="warning",
                    metric_name=stat_name,
                    metric_value=current_val,
                    threshold=anomaly_result.get("upper_bound"),
                    message=(
                        f"{stat_name}={current_val:.1f} выбивается из "
                        f"CI [{anomaly_result['lower_bound']:.1f}, "
                        f"{anomaly_result['upper_bound']:.1f}]"
                    ),
                )

        data_drift = calculate_data_drift(
            reference_features=reference_features,
            current_features=current_features,
            drift_threshold=self.params["monitoring"]["drift_threshold"],
        )

        report_path = ""
        if data_drift["is_drift_detected"]:
            report_path = generate_drift_report(
                reference_df=reference_features,
                current_df=current_features,
                reports_dir=self.params["monitoring"]["reports_dir"],
                data_date=data_date,
            )
            self._create_alert(
                data_date=data_date,
                alert_type="data_drift",
                severity="critical",
                metric_name="drift_score",
                metric_value=data_drift["drift_score"],
                threshold=self.params["monitoring"]["drift_threshold"],
                message=(
                    f"Data drift detected for {data_date}: "
                    f"score={data_drift['drift_score']:.3f}, "
                    f"{data_drift.get('n_features_drifted', '?')} features drifted"
                ),
            )

        self._save_drift_report(data_drift, data_date, report_path)

        daily_metrics = evaluate_model_on_day(
            model=model,
            day_df=day_df,
            users_map=users_map,
            items_map=items_map,
            train_matrix=train_matrix,
        )

        concept_drift = {}
        if daily_metrics:
            baseline_metrics = self._get_baseline_metrics()
            concept_drift = calculate_concept_drift(
                current_metrics=daily_metrics,
                baseline_metrics=baseline_metrics,
                threshold_pct=self.params["monitoring"].get(
                    "concept_drift_threshold_pct", 0.1
                ),
            )
            self._save_drift_report(concept_drift, data_date)

            if concept_drift["is_drift_detected"]:
                for metric_name, info in concept_drift.get(
                    "degraded_metrics", {}
                ).items():
                    self._create_alert(
                        data_date=data_date,
                        alert_type="concept_drift",
                        severity="critical",
                        metric_name=metric_name,
                        metric_value=info["current"],
                        threshold=info["baseline"],
                        message=(
                            f"Metric degradation for {data_date}: "
                            f"{metric_name} dropped {info['drop_pct']}% "
                            f"({info['baseline']} → {info['current']})"
                        ),
                    )

        self._save_daily_stats(data_date, stats, daily_metrics)

        try:
            self._log_to_mlflow(
                data_date=data_date,
                stats=stats,
                daily_metrics=daily_metrics,
                drift_results=(
                    [data_drift, concept_drift] if concept_drift else [data_drift]
                ),
            )
        except Exception as e:
            logger.error(f"MLflow logging error: {e}")

        from services.api.models_db import RetrainingJob

        last_job = (
            self.db.query(RetrainingJob)
            .filter(RetrainingJob.status == "success")
            .order_by(RetrainingJob.finished_at.desc())
            .first()
        )
        if last_job and last_job.finished_at:
            delta = datetime.now(timezone.utc) - last_job.finished_at.replace(
                tzinfo=timezone.utc
            )
            minutes_since_retrain = delta.total_seconds() / 60
        else:
            minutes_since_retrain = -1.0

        self._update_prometheus(
            stats=stats,
            daily_metrics=daily_metrics,
            drift_results=(
                [data_drift, concept_drift] if concept_drift else [data_drift]
            ),
            minutes_since_retrain=minutes_since_retrain,
        )

        summary = {
            "data_date": data_date,
            "stats": stats,
            "daily_metrics": daily_metrics,
            "data_drift": data_drift,
            "concept_drift": concept_drift,
            "anomaly_details": anomaly_details,
            "minutes_since_retrain": minutes_since_retrain,
        }
        logger.info(f"Day {data_date} processed: {summary}")
        return summary
