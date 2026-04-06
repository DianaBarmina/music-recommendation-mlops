from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String

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
    drift_type = Column(String)
    drift_score = Column(Float)
    is_drift_detected = Column(Boolean)
    report_path = Column(String, nullable=True)
    details = Column(JSON, nullable=True)


class RetrainingJob(Base):
    __tablename__ = "retraining_jobs"

    id = Column(Integer, primary_key=True, index=True)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    finished_at = Column(DateTime, nullable=True)
    status = Column(String, default="pending")
    triggered_by = Column(String, default="manual")
    metrics_before = Column(JSON, nullable=True)
    metrics_after = Column(JSON, nullable=True)
    error_message = Column(String, nullable=True)
