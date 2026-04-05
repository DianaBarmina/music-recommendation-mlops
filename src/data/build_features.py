from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, save_npz

from src.utils.helpers import get_logger, load_params

logger = get_logger(__name__)


def build_index_maps(
    train_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:

    users = (
        train_df.select("user_id").unique().sort("user_id").with_row_index("user_idx")
    )
    items = (
        train_df.select("song_id").unique().sort("song_id").with_row_index("item_idx")
    )
    return users, items


def encode_pairs(
    df: pl.DataFrame,
    users: pl.DataFrame,
    items: pl.DataFrame,
) -> pl.DataFrame:

    return df.join(users, on="user_id", how="inner").join(
        items, on="song_id", how="inner"
    )


def to_csr(
    encoded_df: pl.DataFrame,
    n_users: int,
    n_items: int,
    val_col: str = "value",
) -> csr_matrix:
    rows = encoded_df["user_idx"].to_numpy()
    cols = encoded_df["item_idx"].to_numpy()
    data = encoded_df[val_col].to_numpy().astype(np.float32)
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)


def main():
    params = load_params()

    interim_path = params["data"]["interim_path"]
    train_matrix_path = params["data"]["train_matrix_path"]
    val_matrix_path = params["data"]["val_matrix_path"]
    test_matrix_path = params["data"]["test_matrix_path"]
    user_mapping_path = params["data"]["user_mapping_path"]
    item_mapping_path = params["data"]["item_mapping_path"]
    reference_path = params["data"]["reference_path"]
    alpha = params["model"]["alpha"]

    for path in [
        train_matrix_path,
        val_matrix_path,
        test_matrix_path,
        user_mapping_path,
        item_mapping_path,
        reference_path,
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading interim dataset...")
    df = pl.read_parquet(interim_path)

    train_df = df.filter(pl.col("split") == "train")
    val_df = df.filter(pl.col("split") == "val")
    test_df = df.filter(pl.col("split") == "test")

    logger.info("Building index maps from train...")
    users_map, items_map = build_index_maps(train_df)
    n_users = users_map.height
    n_items = items_map.height
    logger.info(f"n_users={n_users}, n_items={n_items}")

    users_map.write_parquet(user_mapping_path)
    items_map.write_parquet(item_mapping_path)
    logger.info("Saved user and item mappings")

    train_encoded = (
        encode_pairs(train_df, users_map, items_map)
        .with_columns(
            (pl.col("play_count").cast(pl.Float32).log1p() * alpha).alias("value")
        )
        .select(["user_idx", "item_idx", "value"])
    )
    train_matrix = to_csr(train_encoded, n_users, n_items)
    save_npz(train_matrix_path, train_matrix)
    logger.info(f"Train matrix: {train_matrix.shape}, nnz={train_matrix.nnz}")

    val_encoded = (
        encode_pairs(val_df, users_map, items_map)
        .with_columns(pl.lit(1.0, dtype=pl.Float32).alias("value"))
        .select(["user_idx", "item_idx", "value"])
    )
    val_matrix = to_csr(val_encoded, n_users, n_items)
    save_npz(val_matrix_path, val_matrix)
    logger.info(f"Val matrix: {val_matrix.shape}, nnz={val_matrix.nnz}")

    test_encoded = (
        encode_pairs(test_df, users_map, items_map)
        .with_columns(pl.lit(1.0, dtype=pl.Float32).alias("value"))
        .select(["user_idx", "item_idx", "value"])
    )
    test_matrix = to_csr(test_encoded, n_users, n_items)
    save_npz(test_matrix_path, test_matrix)
    logger.info(f"Test matrix: {test_matrix.shape}, nnz={test_matrix.nnz}")

    train_df.write_parquet(reference_path)
    logger.info(f"Saved reference dataset to {reference_path}")


if __name__ == "__main__":
    main()
