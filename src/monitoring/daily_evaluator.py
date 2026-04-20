import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from src.models.metrics import (
    hit_rate_at_k,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.utils.helpers import get_logger

logger = get_logger(__name__)

EVAL_K = 10


def evaluate_model_on_day(
    model,
    day_df: pl.DataFrame,
    users_map: pl.DataFrame,
    items_map: pl.DataFrame,
    train_matrix: csr_matrix,
) -> dict:
    """
    Оценивает модель на данных одного дня марта.

    Логика:
    - Берём пользователей дня которые есть в train маппинге (known users)
    - Для каждого такого пользователя треки дня = heldout (то что нужно найти)
    - Исключаем треки которых нет в item маппинге (unseen items)
    - Считаем метрики

    Возвращает словарь с метриками или пустой если данных мало.
    """
    if "play_count" not in day_df.columns:
        day_df = day_df.with_columns(pl.lit(1).alias("play_count"))

    user_to_idx = dict(
        zip(
            users_map["user_id"].to_list(),
            users_map["user_idx"].to_list(),
        )
    )
    item_to_idx = dict(
        zip(
            items_map["song_id"].to_list(),
            items_map["item_idx"].to_list(),
        )
    )

    known_users = day_df.filter(pl.col("user_id").is_in(list(user_to_idx.keys())))
    known_pairs = known_users.filter(pl.col("song_id").is_in(list(item_to_idx.keys())))

    if known_pairs.height < 10:
        logger.warning(f"Too few known pairs for evaluation: {known_pairs.height}")
        return {}

    user_items_day = known_pairs.group_by("user_id").agg(
        pl.col("song_id").alias("songs")
    )

    ndcgs, precisions, recalls, hit_rates, mrrs, maps = [], [], [], [], [], []

    for row in user_items_day.iter_rows(named=True):
        user_id = row["user_id"]
        true_songs = row["songs"]

        user_idx = user_to_idx[user_id]
        true_item_indices = np.array(
            [item_to_idx[s] for s in true_songs if s in item_to_idx],
            dtype=np.int64,
        )

        if true_item_indices.size == 0:
            continue

        recs, _ = model.recommend(
            userid=user_idx,
            user_items=train_matrix[user_idx],
            N=EVAL_K,
            filter_already_liked_items=True,
            recalculate_user=False,
        )
        recs = np.asarray(recs, dtype=np.int64)

        ndcgs.append(ndcg_at_k(recs, true_item_indices, EVAL_K))
        precisions.append(precision_at_k(recs, true_item_indices, EVAL_K))
        recalls.append(recall_at_k(recs, true_item_indices, EVAL_K))
        hit_rates.append(hit_rate_at_k(recs, true_item_indices, EVAL_K))
        mrrs.append(mrr_at_k(recs, true_item_indices, EVAL_K))
        maps.append(map_at_k(recs, true_item_indices, EVAL_K))

    n = len(ndcgs)
    if n == 0:
        return {}

    return {
        f"ndcg@{EVAL_K}": round(float(np.mean(ndcgs)), 4),
        f"precision@{EVAL_K}": round(float(np.mean(precisions)), 4),
        f"recall@{EVAL_K}": round(float(np.mean(recalls)), 4),
        f"hit_rate@{EVAL_K}": round(float(np.mean(hit_rates)), 4),
        f"mrr@{EVAL_K}": round(float(np.mean(mrrs)), 4),
        f"map@{EVAL_K}": round(float(np.mean(maps)), 4),
        "n_eval_users": n,
    }
