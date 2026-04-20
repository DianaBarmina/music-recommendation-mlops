from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text

from services.api.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    user_id = Column(String, index=True, nullable=False)
    recommendations = Column(JSON, nullable=False)
    scores = Column(JSON, nullable=False)
    model_version = Column(String, default="v1")
    n_recommendations = Column(Integer)
    is_anomaly = Column(Boolean, default=False)
    anomaly_reason = Column(String, nullable=True)


class DriftReport(Base):
    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    data_date = Column(String, index=True, nullable=True)
    drift_type = Column(String)
    drift_score = Column(Float)
    is_drift_detected = Column(Boolean)
    report_path = Column(String, nullable=True)
    details = Column(JSON, nullable=True)


class DailyStats(Base):
    __tablename__ = "daily_stats"

    id = Column(Integer, primary_key=True, index=True)
    data_date = Column(String, unique=True, index=True, nullable=False)
    n_interactions = Column(Integer)
    n_unique_users = Column(Integer)
    n_unique_items = Column(Integer)
    avg_play_count = Column(Float)
    median_play_count = Column(Float)
    std_play_count = Column(Float)
    ndcg_at_10 = Column(Float, nullable=True)
    hit_rate_at_10 = Column(Float, nullable=True)
    mrr_at_10 = Column(Float, nullable=True)
    precision_at_10 = Column(Float, nullable=True)
    recall_at_10 = Column(Float, nullable=True)
    map_at_10 = Column(Float, nullable=True)
    n_eval_users = Column(Integer, nullable=True)
    is_anomaly = Column(Boolean, default=False)
    anomaly_details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    data_date = Column(String, nullable=True)
    alert_type = Column(String)
    severity = Column(String)
    metric_name = Column(String, nullable=True)
    metric_value = Column(Float, nullable=True)
    threshold = Column(Float, nullable=True)
    message = Column(Text)
    is_resolved = Column(Boolean, default=False)


class RetrainingJob(Base):
    __tablename__ = "retraining_jobs"

    id = Column(Integer, primary_key=True, index=True)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    finished_at = Column(DateTime, nullable=True)
    status = Column(String, default="pending")
    triggered_by = Column(String, default="manual")
    train_window_start = Column(String, nullable=True)
    train_window_end = Column(String, nullable=True)
    metrics_before = Column(JSON, nullable=True)
    metrics_after = Column(JSON, nullable=True)
    error_message = Column(String, nullable=True)
