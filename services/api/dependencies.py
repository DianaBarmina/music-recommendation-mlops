import pickle

import polars as pl
from scipy.sparse import load_npz

from src.utils.helpers import get_logger

logger = get_logger(__name__)


class ModelArtifacts:

    def __init__(self):
        self.model = None
        self.train_matrix = None
        self.users_map: pl.DataFrame | None = None
        self.items_map: pl.DataFrame | None = None
        self.model_version: str = "v1"
        self.is_ready: bool = False

    def load(self, params: dict) -> None:
        model_path = params["model"]["model_path"]
        train_matrix_path = params["data"]["train_matrix_path"]
        user_mapping_path = params["data"]["user_mapping_path"]
        item_mapping_path = params["data"]["item_mapping_path"]

        logger.info("Loading model artifacts...")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.train_matrix = load_npz(train_matrix_path)
        self.users_map = pl.read_parquet(user_mapping_path)
        self.items_map = pl.read_parquet(item_mapping_path)
        self.is_ready = True

        logger.info(
            f"Artifacts loaded: "
            f"{self.users_map.height} users, "
            f"{self.items_map.height} items"
        )

    def reload(self, params: dict) -> None:
        self.is_ready = False
        self.load(params)


artifacts = ModelArtifacts()


def get_artifacts() -> ModelArtifacts:
    return artifacts
