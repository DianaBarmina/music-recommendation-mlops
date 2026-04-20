from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from services.api.database import get_db
from services.api.models_db import Alert, DailyStats, DriftReport

router = APIRouter(prefix="/drift", tags=["monitoring"])


@router.get("/status")
def get_drift_status(db: Session = Depends(get_db)):
    """Текущий статус дрейфа - последний отчет"""
    latest = db.query(DriftReport).order_by(DriftReport.timestamp.desc()).first()
    if not latest:
        return {"status": "no_data", "is_drift_detected": False}

    return {
        "status": "ok",
        "data_date": latest.data_date,
        "drift_type": latest.drift_type,
        "drift_score": latest.drift_score,
        "is_drift_detected": latest.is_drift_detected,
        "timestamp": latest.timestamp.isoformat(),
        "details": latest.details,
    }


@router.get("/reports")
def get_drift_reports(
    limit: int = Query(default=30, ge=1, le=100),
    drift_type: str | None = None,
    db: Session = Depends(get_db),
):
    """История отчётов о дрейфе."""
    query = db.query(DriftReport).order_by(DriftReport.timestamp.desc())
    if drift_type:
        query = query.filter(DriftReport.drift_type == drift_type)
    reports = query.limit(limit).all()

    return [
        {
            "id": r.id,
            "data_date": r.data_date,
            "timestamp": r.timestamp.isoformat(),
            "drift_type": r.drift_type,
            "drift_score": r.drift_score,
            "is_drift_detected": r.is_drift_detected,
            "report_path": r.report_path,
        }
        for r in reports
    ]


@router.get("/daily-stats")
def get_daily_stats(
    limit: int = Query(default=31, ge=1, le=100),
    db: Session = Depends(get_db),
):
    stats = (
        db.query(DailyStats).order_by(DailyStats.data_date.desc()).limit(limit).all()
    )
    return [
        {
            "data_date": s.data_date,
            "n_interactions": s.n_interactions,
            "n_unique_users": s.n_unique_users,
            "n_unique_items": s.n_unique_items,
            "avg_play_count": s.avg_play_count,
            "ndcg_at_10": s.ndcg_at_10,
            "hit_rate_at_10": s.hit_rate_at_10,
            "mrr_at_10": s.mrr_at_10,
            "precision_at_10": s.precision_at_10,
            "recall_at_10": s.recall_at_10,
            "map_at_10": s.map_at_10,
            "n_eval_users": s.n_eval_users,
            "is_anomaly": s.is_anomaly,
        }
        for s in stats
    ]


@router.get("/alerts")
def get_alerts(
    limit: int = Query(default=50, ge=1, le=200),
    unresolved_only: bool = False,
    db: Session = Depends(get_db),
):
    query = db.query(Alert).order_by(Alert.timestamp.desc())
    if unresolved_only:
        query = query.filter(Alert.is_resolved == False)  # noqa: E712
    alerts = query.limit(limit).all()

    return [
        {
            "id": a.id,
            "data_date": a.data_date,
            "timestamp": a.timestamp.isoformat(),
            "alert_type": a.alert_type,
            "severity": a.severity,
            "metric_name": a.metric_name,
            "metric_value": a.metric_value,
            "threshold": a.threshold,
            "message": a.message,
            "is_resolved": a.is_resolved,
        }
        for a in alerts
    ]


@router.post("/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Alert not found")
    alert.is_resolved = True
    db.commit()
    return {"status": "resolved", "alert_id": alert_id}
