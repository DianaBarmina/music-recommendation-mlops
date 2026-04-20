import shutil
import subprocess
import sys
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
    job = db.query(RetrainingJob).filter(RetrainingJob.id == job_id).first()
    if not job:
        return

    try:
        job.status = "running"
        db.commit()

        logger.info("Starting retraining job %s...", job_id)

        dvc_bin = shutil.which("dvc")

        if dvc_bin:
            cmd = [dvc_bin, "repro", "-f", "train", "-f", "evaluate"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            ok = result.returncode == 0
            stderr = result.stderr
        else:
            cmd_train = [sys.executable, "-X", "utf8", "-m", "src.models.train_model"]
            cmd_eval = [sys.executable, "-X", "utf8", "-m", "src.models.evaluate_model"]

            r1 = subprocess.run(cmd_train, capture_output=True, text=True, timeout=3600)
            if r1.returncode != 0:
                ok = False
                stderr = r1.stderr
            else:
                r2 = subprocess.run(
                    cmd_eval, capture_output=True, text=True, timeout=3600
                )
                ok = r2.returncode == 0
                stderr = r2.stderr

        if ok:
            job.status = "success"
            job.finished_at = datetime.now(timezone.utc)
            db.commit()

            params = load_params()
            artifacts.reload(params)  # перезагрузить модель в памяти API
            logger.info("Retraining job %s completed successfully", job_id)
        else:
            job.status = "failed"
            job.error_message = (stderr or "")[:1000]
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.error("Retraining job %s failed: %s", job_id, stderr)

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
        logger.error("Retraining job %s exception: %s", job_id, e)


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


@router.post("/process-day")
def process_march_day(
    data_date: str,
    day_parquet_path: str,
    db: Session = Depends(get_db),
):
    """
    Ручной запуск обработки одного дня марта.
    Используется для разработки и тестирования мониторинга.
    data_date: например '2007-03-01'
    day_parquet_path: например 'data/raw/march/march_01.parquet'
    """
    from src.monitoring.metrics_collector import PROMETHEUS_METRICS
    from src.monitoring.scheduler import DailyMonitor

    monitor = DailyMonitor(db_session=db, prometheus_metrics=PROMETHEUS_METRICS)
    try:
        summary = monitor.process_day(
            data_date=data_date,
            day_parquet_path=day_parquet_path,
        )
        return {"status": "ok", "summary": summary}
    except Exception as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=str(e))
