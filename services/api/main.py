import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import polars as pl
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from services.api.database import Base, engine
from services.api.dependencies import artifacts
from services.api.models_db import DailyStats
from services.api.routers import health, monitoring_router, recommendations, retraining
from src.monitoring.metrics_collector import PROMETHEUS_METRICS
from src.utils.helpers import get_logger, load_params

logger = get_logger(__name__)


def infer_data_date(parquet_file: Path) -> str:
    min_ts = (
        pl.scan_parquet(str(parquet_file))
        .select(pl.col("ts").min().alias("min_ts"))
        .collect()
        .item()
    )
    return str(min_ts.date())


async def scheduled_march_processing():
    """
    Фоновая задача: каждые 24 часа ищет новые файлы марта и обрабатывает их
    """
    from services.api.database import SessionLocal
    from src.monitoring.scheduler import DailyMonitor

    params = load_params()
    march_dir = Path(params["data"]["raw_march_dir"])
    # processed_dates: set[str] = set()

    check_interval_hours = params["monitoring"].get("check_interval_hours", 24)
    check_interval_seconds = int(check_interval_hours * 3600)

    while True:
        # await asyncio.sleep(86400)
        try:
            db = SessionLocal()
            monitor = DailyMonitor(
                db_session=db,
                prometheus_metrics=PROMETHEUS_METRICS,
            )

            for parquet_file in sorted(march_dir.glob("march_*.parquet")):
                data_date = infer_data_date(parquet_file)

                already_done = (
                    db.query(DailyStats.id)
                    .filter(DailyStats.data_date == data_date)
                    .first()
                    is not None
                )
                if already_done:
                    continue

                try:
                    monitor.process_day(
                        data_date=data_date,
                        day_parquet_path=str(parquet_file),
                    )
                except Exception as e:
                    logger.error(f"Error processing {data_date}: {e}")

            db.close()
        except Exception as e:
            logger.error(f"Scheduler iteration error: {e}")

        await asyncio.sleep(check_interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)

    logger.info("Loading model artifacts...")
    params = load_params()
    artifacts.load(params)

    # task = asyncio.create_task(scheduled_march_processing())

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Music Recommendation API",
    description="ALS-based music recommendation service",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

app.include_router(health.router)
app.include_router(recommendations.router)
app.include_router(retraining.router)
app.include_router(monitoring_router.router)
