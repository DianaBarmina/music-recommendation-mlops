import pickle
from pathlib import Path

from implicit.als import AlternatingLeastSquares
from scipy.sparse import load_npz

from src.utils.helpers import get_logger, load_params

logger = get_logger(__name__)


def train_als(train_matrix, params: dict) -> AlternatingLeastSquares:
    """Обучает ALS модель и возвращает её."""
    model = AlternatingLeastSquares(
        factors=params["model"]["factors"],
        regularization=params["model"]["regularization"],
        iterations=params["model"]["iterations"],
        random_state=params["model"]["random_state"],
        calculate_training_loss=True,
        num_threads=params["model"]["num_threads"],
        use_gpu=False,
    )
    logger.info("Training ALS model...")
    model.fit(train_matrix, show_progress=True)
    return model


def save_model(model, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")


def main():
    params = load_params()

    train_matrix_path = params["data"]["train_matrix_path"]
    model_path = params["model"]["model_path"]

    logger.info(f"Loading train matrix from {train_matrix_path}")
    train_matrix = load_npz(train_matrix_path)

    model = train_als(train_matrix, params)
    save_model(model, model_path)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
