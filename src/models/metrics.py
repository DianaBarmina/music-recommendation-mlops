import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def ndcg_at_k(recs: np.ndarray, true_items: np.ndarray, k: int) -> float:
    recs = np.asarray(recs[:k], dtype=np.int64)
    hits = np.isin(recs, true_items)
    denom = np.log2(np.arange(2, k + 2))
    dcg = (hits / denom).sum()
    m = min(true_items.size, k)
    idcg = (1.0 / denom[:m]).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def precision_at_k(recs: np.ndarray, true_items: np.ndarray, k: int) -> float:
    recs = np.asarray(recs[:k], dtype=np.int64)
    hits = np.isin(recs, true_items)
    return float(hits.mean())


def recall_at_k(recs: np.ndarray, true_items: np.ndarray, k: int) -> float:
    if true_items.size == 0:
        return 0.0
    recs = np.asarray(recs[:k], dtype=np.int64)
    hits = np.isin(recs, true_items)
    return float(hits.sum() / true_items.size)


def hit_rate_at_k(recs: np.ndarray, true_items: np.ndarray, k: int) -> float:
    recs = np.asarray(recs[:k], dtype=np.int64)
    hits = np.isin(recs, true_items)
    return float(hits.any())


def mrr_at_k(recs: np.ndarray, true_items: np.ndarray, k: int) -> float:
    recs = np.asarray(recs[:k], dtype=np.int64)
    hits = np.isin(recs, true_items)
    if hits.any():
        first_hit_pos = int(np.argmax(hits)) + 1
        return 1.0 / first_hit_pos
    return 0.0


def map_at_k(recs: np.ndarray, true_items: np.ndarray, k: int) -> float:
    recs = np.asarray(recs[:k], dtype=np.int64)
    hits = np.isin(recs, true_items)
    if not hits.any():
        return 0.0
    hit_positions = np.flatnonzero(hits)
    precisions_at_hits = np.cumsum(hits)[hit_positions] / (hit_positions + 1)
    return float(precisions_at_hits.mean())


def eval_at_k(
    model,
    train_user_items: csr_matrix,
    heldout_user_items: csr_matrix,
    k: int,
) -> dict:

    n_users = train_user_items.shape[0]

    ndcgs, precisions, recalls, hit_rates, mrrs, maps = [], [], [], [], [], []

    for u in tqdm(range(n_users), desc=f"Eval ALS@{k}", unit="user"):
        true_items = heldout_user_items[u].indices
        if true_items.size == 0:
            continue

        recs, _ = model.recommend(
            userid=u,
            user_items=train_user_items[u],
            N=k,
            filter_already_liked_items=True,
            recalculate_user=True,
        )
        recs = np.asarray(recs, dtype=np.int64)

        ndcgs.append(ndcg_at_k(recs, true_items, k))
        precisions.append(precision_at_k(recs, true_items, k))
        recalls.append(recall_at_k(recs, true_items, k))
        hit_rates.append(hit_rate_at_k(recs, true_items, k))
        mrrs.append(mrr_at_k(recs, true_items, k))
        maps.append(map_at_k(recs, true_items, k))

    n = len(ndcgs)
    return {
        f"ndcg@{k}": float(np.mean(ndcgs)) if n else 0.0,
        f"precision@{k}": float(np.mean(precisions)) if n else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if n else 0.0,
        f"hit_rate@{k}": float(np.mean(hit_rates)) if n else 0.0,
        f"mrr@{k}": float(np.mean(mrrs)) if n else 0.0,
        f"map@{k}": float(np.mean(maps)) if n else 0.0,
        "n_eval_users": n,
    }


def eval_from_recs(
    heldout_user_items: csr_matrix,
    recs_by_user: list[np.ndarray],
    k: int,
    name: str = "",
) -> dict:
    """
    Считает все метрики @ k для заранее построенных рекомендаций.
    Используется для бейзлайнов (popular, random).
    """
    n_users = heldout_user_items.shape[0]

    ndcgs, precisions, recalls, hit_rates, mrrs, maps = [], [], [], [], [], []

    for u in tqdm(range(n_users), desc=f"Eval {name}@{k}", unit="user"):
        true_items = heldout_user_items[u].indices
        if true_items.size == 0:
            continue

        recs = np.asarray(recs_by_user[u][:k], dtype=np.int64)

        ndcgs.append(ndcg_at_k(recs, true_items, k))
        precisions.append(precision_at_k(recs, true_items, k))
        recalls.append(recall_at_k(recs, true_items, k))
        hit_rates.append(hit_rate_at_k(recs, true_items, k))
        mrrs.append(mrr_at_k(recs, true_items, k))
        maps.append(map_at_k(recs, true_items, k))

    n = len(ndcgs)
    return {
        f"ndcg@{k}": float(np.mean(ndcgs)) if n else 0.0,
        f"precision@{k}": float(np.mean(precisions)) if n else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if n else 0.0,
        f"hit_rate@{k}": float(np.mean(hit_rates)) if n else 0.0,
        f"mrr@{k}": float(np.mean(mrrs)) if n else 0.0,
        f"map@{k}": float(np.mean(maps)) if n else 0.0,
        "n_eval_users": n,
    }
