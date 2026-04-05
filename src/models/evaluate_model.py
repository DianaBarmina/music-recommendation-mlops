import pickle

import numpy as np
from scipy.sparse import csr_matrix, load_npz
from tqdm import tqdm

from src.utils.helpers import get_logger, load_params, save_metrics

logger = get_logger(__name__)


def eval_at_k(
    model,
    train_user_items: csr_matrix,
    heldout_user_items: csr_matrix,
    k: int,
) -> dict:
    n_users = train_user_items.shape[0]
    precisions, recalls, maps, ndcgs, hits_list = [], [], [], [], []

    for u in tqdm(range(n_users), desc=f"Eval@{k}", unit="user"):
        true_items = heldout_user_items[u].indices
        if true_items.size == 0:
            continue

        recs, _scores = model.recommend(
            userid=u,
            user_items=train_user_items[u],
            N=k,
            filter_already_liked_items=True,
            recalculate_user=True,
        )
        recs = np.asarray(recs, dtype=np.int64)
        hits = np.isin(recs, true_items)

        precisions.append(hits.mean())
        recalls.append(hits.sum() / true_items.size)
        hits_list.append(float(hits.any()))

        if hits.any():
            hit_idx = np.flatnonzero(hits)
            prec_at_i = np.cumsum(hits)[hit_idx] / (hit_idx + 1)
            maps.append(prec_at_i.mean())
        else:
            maps.append(0.0)

        denom = np.log2(np.arange(2, k + 2))
        dcg = (hits / denom).sum()
        m = min(true_items.size, k)
        idcg = (1.0 / denom[:m]).sum()
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    # MRR
    mrr_list = []
    for u in range(n_users):
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
        hits = np.isin(recs, true_items)
        if hits.any():
            first_hit = np.argmax(hits) + 1
            mrr_list.append(1.0 / first_hit)
        else:
            mrr_list.append(0.0)

    n = len(precisions)
    return {
        f"precision@{k}": float(np.mean(precisions)) if n else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if n else 0.0,
        f"map@{k}": float(np.mean(maps)) if n else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if n else 0.0,
        f"hit_rate@{k}": float(np.mean(hits_list)) if n else 0.0,
        f"mrr@{k}": float(np.mean(mrr_list)) if mrr_list else 0.0,
        "n_eval_users": n,
    }


def eval_from_recs(
    heldout_user_items: csr_matrix,
    recs_by_user: list,
    k: int,
    name: str = "",
) -> dict:
    precisions, recalls, maps, ndcgs, hits_list, mrr_list = [], [], [], [], [], []
    n_users = heldout_user_items.shape[0]

    for u in tqdm(range(n_users), desc=f"Eval {name}@{k}", unit="user"):
        true_items = heldout_user_items[u].indices
        if true_items.size == 0:
            continue

        recs = np.asarray(recs_by_user[u][:k], dtype=np.int64)
        hits = np.isin(recs, true_items)

        precisions.append(hits.mean())
        recalls.append(hits.sum() / true_items.size)
        hits_list.append(float(hits.any()))

        if hits.any():
            hit_idx = np.flatnonzero(hits)
            prec_at_i = np.cumsum(hits)[hit_idx] / (hit_idx + 1)
            maps.append(prec_at_i.mean())
            first_hit = np.argmax(hits) + 1
            mrr_list.append(1.0 / first_hit)
        else:
            maps.append(0.0)
            mrr_list.append(0.0)

        denom = np.log2(np.arange(2, k + 2))
        dcg = (hits / denom).sum()
        m = min(true_items.size, k)
        idcg = (1.0 / denom[:m]).sum()
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    n = len(precisions)
    return {
        f"precision@{k}": float(np.mean(precisions)) if n else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if n else 0.0,
        f"map@{k}": float(np.mean(maps)) if n else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if n else 0.0,
        f"hit_rate@{k}": float(np.mean(hits_list)) if n else 0.0,
        f"mrr@{k}": float(np.mean(mrr_list)) if n else 0.0,
        "n_eval_users": n,
    }


def build_popular_recs(train_user_items: csr_matrix, max_k: int) -> list:
    item_pop = np.asarray(train_user_items.sum(axis=0)).ravel()
    popular_items = np.argsort(-item_pop)
    recs_by_user = []
    for u in range(train_user_items.shape[0]):
        seen = train_user_items[u].indices
        mask = ~np.isin(popular_items, seen)
        recs_by_user.append(popular_items[mask][:max_k])
    return recs_by_user


def build_random_recs(train_user_items: csr_matrix, max_k: int, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    n_users, n_items = train_user_items.shape
    all_items = np.arange(n_items, dtype=np.int64)
    recs_by_user = []
    for u in range(n_users):
        seen = train_user_items[u].indices
        candidates = np.setdiff1d(all_items, seen)
        if candidates.size <= max_k:
            recs_by_user.append(candidates)
        else:
            recs_by_user.append(rng.choice(candidates, size=max_k, replace=False))
    return recs_by_user


def main():
    params = load_params()

    model_path = params["model"]["model_path"]
    train_matrix_path = params["data"]["train_matrix_path"]
    val_matrix_path = params["data"]["val_matrix_path"]
    test_matrix_path = params["data"]["test_matrix_path"]
    ks = params["metrics"]["ks"]

    logger.info("Loading model and matrices...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    train_matrix = load_npz(train_matrix_path)
    val_matrix = load_npz(val_matrix_path)
    test_matrix = load_npz(test_matrix_path)

    logger.info(
        f"Shapes: train={train_matrix.shape}, "
        f"val={val_matrix.shape}, test={test_matrix.shape}"
    )

    MAX_K = max(ks)
    popular_recs = build_popular_recs(train_matrix, max_k=MAX_K)
    random_recs = build_random_recs(
        train_matrix, max_k=MAX_K, seed=params["model"]["random_state"]
    )

    all_metrics = {}

    for k in ks:
        logger.info(f"\n{'='*40}\nEvaluating K={k}")

        als_val = eval_at_k(model, train_matrix, val_matrix, k=k)
        als_test = eval_at_k(model, train_matrix, test_matrix, k=k)
        pop_val = eval_from_recs(val_matrix, popular_recs, k=k, name="popular")
        pop_test = eval_from_recs(test_matrix, popular_recs, k=k, name="popular")
        rnd_val = eval_from_recs(val_matrix, random_recs, k=k, name="random")
        rnd_test = eval_from_recs(test_matrix, random_recs, k=k, name="random")

        logger.info(f"ALS   VAL  @{k}: {als_val}")
        logger.info(f"ALS   TEST @{k}: {als_test}")
        logger.info(f"POP   VAL  @{k}: {pop_val}")
        logger.info(f"POP   TEST @{k}: {pop_test}")
        logger.info(f"RAND  VAL  @{k}: {rnd_val}")
        logger.info(f"RAND  TEST @{k}: {rnd_test}")

        for metric_name, value in als_test.items():
            all_metrics[f"als_test_{metric_name}"] = value
        for metric_name, value in als_val.items():
            all_metrics[f"als_val_{metric_name}"] = value
        for metric_name, value in pop_test.items():
            all_metrics[f"popular_test_{metric_name}"] = value
        for metric_name, value in rnd_test.items():
            all_metrics[f"random_test_{metric_name}"] = value

    save_metrics(all_metrics, "metrics.json")
    logger.info("Metrics saved to metrics.json")


if __name__ == "__main__":
    main()
