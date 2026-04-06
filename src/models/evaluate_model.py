import pickle
import re
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from scipy.sparse import csr_matrix, load_npz

from src.models.metrics import eval_at_k, eval_from_recs
from src.utils.helpers import get_logger, load_params, save_metrics

logger = get_logger(__name__)

_ALLOWED = re.compile(r"[^0-9a-zA-Z_\-./ ]+")


def sanitize_mlflow_metric_name(name: str) -> str:
    name = name.replace("@", "_at_")
    name = _ALLOWED.sub("_", name)
    return name


def sanitize_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {sanitize_mlflow_metric_name(k): float(v) for k, v in metrics.items()}


def build_popular_recs(train_user_items: csr_matrix, max_k: int) -> list:
    item_pop = np.asarray(train_user_items.sum(axis=0)).ravel()
    popular_items = np.argsort(-item_pop)
    recs_by_user = []
    for u in range(train_user_items.shape[0]):
        seen = train_user_items[u].indices
        mask = ~np.isin(popular_items, seen)
        recs_by_user.append(popular_items[mask][:max_k])
    return recs_by_user


def build_random_recs(
    train_user_items: csr_matrix,
    max_k: int,
    seed: int = 42,
) -> list:
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


def load_model(model_path: str) -> Any:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def log_metrics_to_mlflow(params: dict, metrics: dict[str, float]) -> None:
    """
    Логируем метрики в MLflow:
    - если есть run_id от train => логируем в него
    - иначе => создаём новый run и логируем туда
    """
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    sanitized = sanitize_metrics(metrics)
    run_id_path = Path("models/mlflow_run_id.txt")

    # UTF-8 на всякий случай для Windows-консоли
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if run_id_path.exists():
        run_id = run_id_path.read_text(encoding="utf-8").strip()
        logger.info("Logging metrics to existing MLflow run_id=%s", run_id)
        try:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(sanitized)
                mlflow.log_artifact("metrics.json")
            return
        except Exception as e:
            logger.warning("Failed to log to existing run_id=%s: %s", run_id, e)

    logger.warning("No valid MLflow run_id found, logging metrics to a new run")
    with mlflow.start_run():
        mlflow.log_metrics(sanitized)
        mlflow.log_artifact("metrics.json")


def main() -> None:
    params = load_params()

    model_path = params["model"]["model_path"]
    train_matrix_path = params["data"]["train_matrix_path"]
    val_matrix_path = params["data"]["val_matrix_path"]
    test_matrix_path = params["data"]["test_matrix_path"]
    ks = params["metrics"]["ks"]

    logger.info("Loading model and matrices...")
    model = load_model(model_path)

    train_matrix = load_npz(train_matrix_path).tocsr()
    val_matrix = load_npz(val_matrix_path).tocsr()
    test_matrix = load_npz(test_matrix_path).tocsr()

    logger.info(
        "Shapes: train=%s, val=%s, test=%s",
        train_matrix.shape,
        val_matrix.shape,
        test_matrix.shape,
    )

    max_k = max(ks)
    popular_recs = build_popular_recs(train_matrix, max_k=max_k)
    random_recs = build_random_recs(
        train_matrix,
        max_k=max_k,
        seed=int(params["model"]["random_state"]),
    )

    all_metrics: dict[str, float] = {}

    for k in ks:
        logger.info("%s", "\n" + "=" * 40 + f"\nEvaluating K={k}")

        als_val = eval_at_k(model, train_matrix, val_matrix, k=k)
        als_test = eval_at_k(model, train_matrix, test_matrix, k=k)

        pop_val = eval_from_recs(val_matrix, popular_recs, k=k, name="popular")
        pop_test = eval_from_recs(test_matrix, popular_recs, k=k, name="popular")

        rnd_val = eval_from_recs(val_matrix, random_recs, k=k, name="random")
        rnd_test = eval_from_recs(test_matrix, random_recs, k=k, name="random")

        logger.info("ALS   VAL  @%d: %s", k, als_val)
        logger.info("ALS   TEST @%d: %s", k, als_test)
        logger.info("POP   VAL  @%d: %s", k, pop_val)
        logger.info("POP   TEST @%d: %s", k, pop_test)
        logger.info("RAND  VAL  @%d: %s", k, rnd_val)
        logger.info("RAND  TEST @%d: %s", k, rnd_test)

        for metric_name, value in als_test.items():
            all_metrics[f"als_test_{metric_name}"] = float(value)
        for metric_name, value in als_val.items():
            all_metrics[f"als_val_{metric_name}"] = float(value)

        for metric_name, value in pop_test.items():
            all_metrics[f"popular_test_{metric_name}"] = float(value)
        for metric_name, value in rnd_test.items():
            all_metrics[f"random_test_{metric_name}"] = float(value)

    save_metrics(all_metrics, "metrics.json")
    logger.info("Metrics saved to metrics.json")

    log_metrics_to_mlflow(params, all_metrics)
    logger.info("Metrics logged to MLflow")


if __name__ == "__main__":
    main()
