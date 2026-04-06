import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from services.api.database import get_db
from services.api.dependencies import ModelArtifacts, get_artifacts
from services.api.models_db import Prediction
from src.models.predict_model import get_recommendations, get_user_idx
from src.utils.helpers import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: list[str]
    scores: list[float]
    model_version: str
    timestamp: str


class PredictionRow(BaseModel):
    id: int
    timestamp: str
    user_id: str
    recommendations: list[str]
    scores: list[float]
    model_version: str
    is_anomaly: bool
    anomaly_reason: str | None


@router.get("/{user_id}", response_model=RecommendationResponse)
def recommend(
    user_id: str,
    n_items: int = Query(default=10, ge=1, le=100),
    artifacts: ModelArtifacts = Depends(get_artifacts),
    db: Session = Depends(get_db),
):
    if not artifacts.is_ready:
        raise HTTPException(status_code=503, detail="Model is not ready")

    user_idx = get_user_idx(user_id, artifacts.users_map)
    if user_idx is None:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}' not found in training data",
        )

    songs, scores = get_recommendations(
        user_id=user_id,
        n_items=n_items,
        model=artifacts.model,
        train_matrix=artifacts.train_matrix,
        users_map=artifacts.users_map,
        items_map=artifacts.items_map,
    )

    is_anomaly = len(songs) < n_items
    anomaly_reason = (
        f"Only {len(songs)} recommendations available" if is_anomaly else None
    )

    prediction = Prediction(
        user_id=user_id,
        recommendations=songs,
        scores=scores,
        model_version=artifacts.model_version,
        n_recommendations=len(songs),
        is_anomaly=is_anomaly,
        anomaly_reason=anomaly_reason,
    )
    db.add(prediction)
    db.commit()

    return RecommendationResponse(
        user_id=user_id,
        recommendations=songs,
        scores=scores,
        model_version=artifacts.model_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/history/latest", response_model=list[PredictionRow])
def get_predictions_history(
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Последние N предсказаний для таблицы в UI"""
    predictions = (
        db.query(Prediction).order_by(Prediction.timestamp.desc()).limit(limit).all()
    )
    return [
        PredictionRow(
            id=p.id,
            timestamp=p.timestamp.isoformat(),
            user_id=p.user_id,
            recommendations=p.recommendations,
            scores=p.scores,
            model_version=p.model_version,
            is_anomaly=p.is_anomaly,
            anomaly_reason=p.anomaly_reason,
        )
        for p in predictions
    ]


@router.get("/metrics/current")
def get_current_metrics():
    """Текущие метрики модели из metrics.json"""
    metrics_path = Path("metrics.json")
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found")
    with open(metrics_path) as f:
        return json.load(f)
