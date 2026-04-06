import subprocess
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from services.api.database import get_db
from services.api.dependencies import ModelArtifacts, get_artifacts
from services.api.models_db import RetrainingJob
from src.utils.helpers import get_logger, load_params

logger = get_logger(__name__)
router = APIRouter(prefix="/retrain", tags=["retraining"])


class RetrainingResponse(BaseModel):
    job_id: int
    status: str
    message: str
    started_at: str


def run_retraining(job_id: int, db: Session, artifacts: ModelArtifacts):
    """Запускает dvc repro в фоне и обновляет статус в бд"""
    job = db.query(RetrainingJob).filter(RetrainingJob.id == job_id).first()
    if not job:
        return

    try:
        job.status = "running"
        db.commit()

        logger.info(f"Starting retraining job {job_id}...")
        result = subprocess.run(
            ["dvc", "repro"],
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode == 0:
            job.status = "success"
            job.finished_at = datetime.now(timezone.utc)
            db.commit()

            params = load_params()
            artifacts.reload(params)
            logger.info(f"Retraining job {job_id} completed successfully")
        else:
            job.status = "failed"
            job.error_message = result.stderr[:1000]
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.error(f"Retraining job {job_id} failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        job.status = "failed"
        job.error_message = "Timeout after 3600 seconds"
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        logger.error(f"Retraining job {job_id} exception: {e}")


@router.post("/", response_model=RetrainingResponse)
def trigger_retraining(
    background_tasks: BackgroundTasks,
    triggered_by: str = "manual",
    db: Session = Depends(get_db),
    artifacts: ModelArtifacts = Depends(get_artifacts),
):
    """Запускает переобучение в фоне"""
    job = RetrainingJob(
        status="pending",
        triggered_by=triggered_by,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(run_retraining, job.id, db, artifacts)

    return RetrainingResponse(
        job_id=job.id,
        status="pending",
        message="Retraining started in background",
        started_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/status/latest")
def get_latest_retraining_status(db: Session = Depends(get_db)):
    """Статус последнего переобучения"""
    job = db.query(RetrainingJob).order_by(RetrainingJob.started_at.desc()).first()
    if not job:
        return {"status": "no_jobs", "message": "No retraining jobs found"}

    return {
        "job_id": job.id,
        "status": job.status,
        "triggered_by": job.triggered_by,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "error_message": job.error_message,
    }


@router.get("/history")
def get_retraining_history(
    limit: int = 10,
    db: Session = Depends(get_db),
):
    jobs = (
        db.query(RetrainingJob)
        .order_by(RetrainingJob.started_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "job_id": j.id,
            "status": j.status,
            "triggered_by": j.triggered_by,
            "started_at": j.started_at.isoformat() if j.started_at else None,
            "finished_at": j.finished_at.isoformat() if j.finished_at else None,
        }
        for j in jobs
    ]
