import pickle
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from scipy.sparse import csr_matrix

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def load_model(model_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")
    return model


def load_mappings(
    user_mapping_path: str,
    item_mapping_path: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    users_map = pl.read_parquet(user_mapping_path)
    items_map = pl.read_parquet(item_mapping_path)
    return users_map, items_map


def get_user_idx(user_id: str, users_map: pl.DataFrame) -> int | None:
    result = users_map.filter(pl.col("user_id") == user_id)
    if result.is_empty():
        return None
    return result["user_idx"][0]


def get_recommendations(
    user_id: str,
    n_items: int,
    model,
    train_matrix: csr_matrix,
    users_map: pl.DataFrame,
    items_map: pl.DataFrame,
) -> tuple[list[str], list[float]]:

    user_idx = get_user_idx(user_id, users_map)
    if user_idx is None:
        logger.warning(f"Unknown user_id: {user_id}")
        return [], []

    user_row = train_matrix[user_idx]
    recs_idx, scores = model.recommend(
        userid=user_idx,
        user_items=user_row,
        N=n_items,
        filter_already_liked_items=True,
        recalculate_user=False,
    )

    idx_to_song = dict(
        zip(items_map["item_idx"].to_list(), items_map["song_id"].to_list())
    )
    song_ids = [idx_to_song[int(idx)] for idx in recs_idx]
    scores_list = [float(s) for s in scores]

    return song_ids, scores_list


def log_prediction(
    user_id: str,
    recommendations: list[str],
    scores: list[float],
    predictions_log_path: str,
    model_version: str = "v1",
) -> None:

    log_path = Path(predictions_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pl.DataFrame(
        {
            "timestamp": [datetime.now(timezone.utc).isoformat()],
            "user_id": [user_id],
            "recommendations": [recommendations],
            "scores": [scores],
            "model_version": [model_version],
            "n_recommendations": [len(recommendations)],
        }
    )

    if log_path.exists():
        existing = pl.read_parquet(str(log_path))
        updated = pl.concat([existing, new_row], how="diagonal")
    else:
        updated = new_row

    updated.write_parquet(str(log_path))
