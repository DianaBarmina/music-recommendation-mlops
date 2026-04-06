from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from services.api.database import Base, engine
from services.api.dependencies import artifacts
from services.api.routers import health, monitoring_router, recommendations, retraining
from src.utils.helpers import get_logger, load_params

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)

    logger.info("Loading model artifacts...")
    params = load_params()
    artifacts.load(params)

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
