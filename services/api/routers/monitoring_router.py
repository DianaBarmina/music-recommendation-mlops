from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from services.api.database import get_db
from services.api.models_db import DriftReport

router = APIRouter(prefix="/drift", tags=["monitoring"])


@router.get("/status")
def get_drift_status(db: Session = Depends(get_db)):
    """Текущий статус дрейфа - последний отчет"""
    latest = db.query(DriftReport).order_by(DriftReport.timestamp.desc()).first()
    if not latest:
        return {
            "status": "no_data",
            "is_drift_detected": False,
            "message": "No drift reports yet",
        }

    return {
        "status": "ok",
        "drift_type": latest.drift_type,
        "drift_score": latest.drift_score,
        "is_drift_detected": latest.is_drift_detected,
        "timestamp": latest.timestamp.isoformat(),
        "details": latest.details,
    }


@router.get("/reports")
def get_drift_reports(limit: int = 20, db: Session = Depends(get_db)):
    reports = (
        db.query(DriftReport).order_by(DriftReport.timestamp.desc()).limit(limit).all()
    )
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "drift_type": r.drift_type,
            "drift_score": r.drift_score,
            "is_drift_detected": r.is_drift_detected,
            "report_path": r.report_path,
        }
        for r in reports
    ]
