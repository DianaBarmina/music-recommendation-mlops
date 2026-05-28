import shutil
import subprocess
import sys
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from services.api.database import SessionLocal, get_db
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


def _get_new_db() -> Session:
    """
    Создаёт новую независимую сессию БД.
    Используется в фоновых задачах где сессия из запроса уже закрыта.
    """
    return SessionLocal()


def run_retraining(
    job_id: int,
    artifacts: ModelArtifacts,
    current_date: str | None = None,
) -> None:
    """
    Фоновая задача переобучения.

    Создаёт собственную сессию БД — не принимает сессию из запроса,
    потому что та закрывается сразу после ответа клиенту.

    Если current_date передана — собирает rolling window и обучает на нём.
    Если нет — обучает на исходных данных (январь+февраль).
    """
    db = _get_new_db()
    try:
        job = db.query(RetrainingJob).filter(RetrainingJob.id == job_id).first()
        if not job:
            logger.error("Job %d not found in DB", job_id)
            return

        job.status = "running"
        db.commit()
        logger.info(
            "Starting retraining job %d (current_date=%s)", job_id, current_date
        )

        params = load_params()
        env_extra: dict[str, str] = {}

        # Если есть дата — строим rolling window и передаём путь через env
        if current_date:
            from src.monitoring.drift_detector import build_retrain_window

            retrain_window_path = params["data"].get(
                "retrain_window_path",
                "data/interim/retrain_window.parquet",
            )
            try:
                window_start, window_end = build_retrain_window(
                    current_date=current_date,
                    march_dir=params["data"]["raw_march_dir"],
                    january_path=params["data"]["january_path"],
                    february_path=params["data"]["february_path"],
                    output_path=retrain_window_path,
                    window_days=params["monitoring"].get("rolling_window_days", 45),
                )
                env_extra["RETRAIN_DATA_PATH"] = retrain_window_path
                job.train_window_start = window_start
                job.train_window_end = window_end
                db.commit()
                logger.info(
                    "Retrain window built: %s — %s (%s)",
                    window_start,
                    window_end,
                    retrain_window_path,
                )
            except Exception as e:
                logger.error("Failed to build retrain window: %s", e)
                job.status = "failed"
                job.error_message = f"Window build error: {e}"
                job.finished_at = datetime.now(timezone.utc)
                db.commit()
                return

        # Запускаем обучение
        import os

        run_env = {**os.environ, **env_extra}
        ok, stderr = _run_pipeline(run_env)

        if ok:
            job.status = "success"
            job.finished_at = datetime.now(timezone.utc)
            db.commit()

            # Перезагружаем модель в памяти сервиса
            artifacts.reload(params)
            logger.info("Retraining job %d completed successfully", job_id)
        else:
            job.status = "failed"
            job.error_message = (stderr or "")[:1000]
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.error("Retraining job %d failed: %s", job_id, stderr)

    except Exception as e:
        try:
            job = db.query(RetrainingJob).filter(RetrainingJob.id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = str(e)[:1000]
                job.finished_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            pass
        logger.error("Retraining job %d exception: %s", job_id, e)
    finally:
        db.close()


def _run_pipeline(env: dict) -> tuple[bool, str]:
    """
    Запускает пайплайн обучения.
    Сначала пробует dvc, если не найден — запускает скрипты напрямую.
    Возвращает (success, stderr).
    """
    dvc_bin = shutil.which("dvc")

    if dvc_bin:
        result = subprocess.run(
            [dvc_bin, "repro", "--force", "train", "evaluate"],
            capture_output=True,
            text=True,
            timeout=3600,
            env=env,
        )
        return result.returncode == 0, result.stderr

    # Fallback: запускаем скрипты напрямую
    for module in (
        "src.data.make_dataset",
        "src.data.build_features",
        "src.models.train_model",
        "src.models.evaluate_model",
    ):
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", module],
            capture_output=True,
            text=True,
            timeout=3600,
            env=env,
        )
        if result.returncode != 0:
            logger.error("Module %s failed: %s", module, result.stderr)
            return False, result.stderr

    return True, ""


@router.post("/", response_model=RetrainingResponse)
def trigger_retraining(
    background_tasks: BackgroundTasks,
    triggered_by: str = "manual",
    current_date: str | None = None,
    db: Session = Depends(get_db),
    artifacts: ModelArtifacts = Depends(get_artifacts),
):
    """
    Запускает переобучение в фоне.

    current_date (опционально): дата в формате YYYY-MM-DD.
    Если передана — переобучение на rolling window последних 45 дней.
    Если не передана — переобучение на исходных данных (январь+февраль).
    """
    job = RetrainingJob(
        status="pending",
        triggered_by=triggered_by,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Передаём только job_id и current_date — НЕ передаём db сессию
    background_tasks.add_task(
        run_retraining,
        job.id,
        artifacts,
        current_date,
    )

    return RetrainingResponse(
        job_id=job.id,
        status="pending",
        message=(
            "Retraining started in background"
            + (f" with rolling window up to {current_date}" if current_date else "")
        ),
        started_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/status/latest")
def get_latest_retraining_status(db: Session = Depends(get_db)):
    job = db.query(RetrainingJob).order_by(RetrainingJob.started_at.desc()).first()
    if not job:
        return {"status": "no_jobs", "message": "No retraining jobs found"}

    return {
        "job_id": job.id,
        "status": job.status,
        "triggered_by": job.triggered_by,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "train_window_start": job.train_window_start,
        "train_window_end": job.train_window_end,
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
            "train_window_start": j.train_window_start,
            "train_window_end": j.train_window_end,
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
        # Конвертируем numpy типы в Python чтобы FastAPI мог сериализовать
        return {"status": "ok", "summary": _make_serializable(summary)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _make_serializable(obj):
    """Рекурсивно конвертирует numpy типы в стандартные Python типы."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
