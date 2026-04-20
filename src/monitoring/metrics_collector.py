from prometheus_client import Counter, Gauge, Histogram

RECOMMENDATIONS_TOTAL = Counter(
    "recommendations_total",
    "Total recommendation requests",
    ["model_version"],
)

RECOMMENDATIONS_LATENCY = Histogram(
    "recommendations_latency_seconds",
    "Recommendation latency",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)

ANOMALIES_TOTAL = Counter(
    "anomalies_total",
    "Total anomaly flags",
    ["reason"],
)

DRIFT_SCORE = Gauge(
    "drift_score",
    "Current drift score",
    ["drift_type"],
)

DRIFT_DETECTED = Gauge(
    "drift_detected",
    "1 if drift detected",
    ["drift_type"],
)

MODEL_METRIC = Gauge(
    "model_metric",
    "Model quality metric",
    ["metric_name", "split"],
)

RETRAINING_TOTAL = Counter(
    "retraining_total",
    "Retraining jobs count",
    ["status"],
)

MINUTES_SINCE_RETRAIN = Gauge(
    "minutes_since_last_retrain",
    "Minutes elapsed since last successful retraining",
)

DATA_N_INTERACTIONS = Gauge(
    "data_n_interactions_daily",
    "Number of interactions in the latest processed day",
)

DATA_N_USERS = Gauge(
    "data_n_unique_users_daily",
    "Number of unique users in the latest processed day",
)

ALERTS_TOTAL = Counter(
    "alerts_total",
    "Total alerts fired",
    ["alert_type", "severity"],
)

PROMETHEUS_METRICS = {
    "DRIFT_SCORE": DRIFT_SCORE,
    "DRIFT_DETECTED": DRIFT_DETECTED,
    "MODEL_METRIC": MODEL_METRIC,
    "MINUTES_SINCE_RETRAIN": MINUTES_SINCE_RETRAIN,
    "DATA_N_INTERACTIONS": DATA_N_INTERACTIONS,
    "DATA_N_USERS": DATA_N_USERS,
    "ALERTS_TOTAL": ALERTS_TOTAL,
}
