import os
from datetime import timedelta
from pathlib import Path

import polars as pl

from src.utils.helpers import get_logger, load_params

logger = get_logger(__name__)


def load_raw_data(january_path: str, february_path: str) -> pl.LazyFrame:
    jan = pl.scan_parquet(january_path).select(["user_id", "song_id", "ts"])
    feb = pl.scan_parquet(february_path).select(["user_id", "song_id", "ts"])
    return pl.concat([jan, feb])


def load_retrain_data(retrain_window_path: str) -> pl.LazyFrame:
    """Загружает данные rolling window для переобучения."""
    return pl.scan_parquet(retrain_window_path).select(["user_id", "song_id", "ts"])


def filter_cold_users_and_items(
    train_pairs: pl.LazyFrame,
    min_user_items: int,
    min_item_users: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.LazyFrame]:
    good_users = (
        train_pairs.group_by("user_id")
        .agg(pl.len().alias("n_items"))
        .filter(pl.col("n_items") >= min_user_items)
        .select("user_id")
    )
    good_items = (
        train_pairs.group_by("song_id")
        .agg(pl.len().alias("n_users"))
        .filter(pl.col("n_users") >= min_item_users)
        .select("song_id")
    )
    filtered = train_pairs.join(good_users, on="user_id", how="inner").join(
        good_items, on="song_id", how="inner"
    )
    return good_users.collect(), good_items.collect(), filtered


def split_by_time_window(
    lf: pl.LazyFrame,
    good_users: pl.DataFrame,
    good_items: pl.DataFrame,
    train_pairs: pl.LazyFrame,
    test_window_days: int,
    val_window_days: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    max_ts = lf.select(pl.col("ts").max()).collect().item()
    test_start = max_ts - timedelta(days=test_window_days)
    val_start = test_start - timedelta(days=val_window_days)

    logger.info("Max date: %s", max_ts)
    logger.info("Train: до %s", val_start)
    logger.info("Val:   %s — %s", val_start, test_start)
    logger.info("Test:  %s — %s", test_start, max_ts)

    val_raw = lf.filter(
        (pl.col("ts") >= pl.lit(val_start)) & (pl.col("ts") < pl.lit(test_start))
    )
    test_raw = lf.filter(pl.col("ts") >= pl.lit(test_start))

    val_pairs = (
        val_raw.select(["user_id", "song_id"])
        .unique()
        .join(good_users.lazy(), on="user_id", how="inner")
        .join(good_items.lazy(), on="song_id", how="inner")
        .join(train_pairs, on=["user_id", "song_id"], how="anti")
        .collect()
    )
    test_pairs = (
        test_raw.select(["user_id", "song_id"])
        .unique()
        .join(good_users.lazy(), on="user_id", how="inner")
        .join(good_items.lazy(), on="song_id", how="inner")
        .join(train_pairs, on=["user_id", "song_id"], how="anti")
        .collect()
    )

    train_pairs_collected = train_pairs.collect()

    dates_info = pl.DataFrame(
        {
            "val_start": [val_start],
            "test_start": [test_start],
            "max_ts": [max_ts],
        }
    )

    return train_pairs_collected, val_pairs, test_pairs, dates_info


def main() -> None:
    params = load_params()

    # Если задана переменная окружения — режим переобучения на rolling window
    retrain_window_path = os.getenv("RETRAIN_DATA_PATH", "")

    january_path = params["data"]["january_path"]
    february_path = params["data"]["february_path"]
    interim_path = params["data"]["interim_path"]
    min_user_items = params["split"]["min_user_items"]
    min_item_users = params["split"]["min_item_users"]
    test_window_days = params["split"]["test_window_days"]
    val_window_days = params["split"]["val_window_days"]

    Path(interim_path).parent.mkdir(parents=True, exist_ok=True)

    if retrain_window_path and Path(retrain_window_path).exists():
        logger.info("RETRAIN MODE: loading rolling window from %s", retrain_window_path)
        lf = load_retrain_data(retrain_window_path)
    else:
        logger.info("INITIAL TRAIN MODE: loading january + february")
        lf = load_raw_data(january_path, february_path)

    total_window = test_window_days + val_window_days
    max_ts = lf.select(pl.col("ts").max()).collect().item()
    train_cutoff = max_ts - timedelta(days=total_window)

    train_raw_pairs = (
        lf.filter(pl.col("ts") < pl.lit(train_cutoff))
        .select(["user_id", "song_id"])
        .unique()
    )

    logger.info("Filtering cold users and items...")
    good_users, good_items, train_pairs_filtered = filter_cold_users_and_items(
        train_raw_pairs, min_user_items, min_item_users
    )

    logger.info(
        "After filtering: %d users, %d items",
        good_users.height,
        good_items.height,
    )

    train_df, val_df, test_df, dates_info = split_by_time_window(
        lf,
        good_users,
        good_items,
        train_pairs_filtered,
        test_window_days,
        val_window_days,
    )

    train_df = train_df.with_columns(pl.lit("train").alias("split"))
    val_df = val_df.with_columns(pl.lit("val").alias("split"))
    test_df = test_df.with_columns(pl.lit("test").alias("split"))

    combined = pl.concat([train_df, val_df, test_df], how="diagonal")

    play_counts = (
        lf.group_by(["user_id", "song_id"]).agg(pl.len().alias("play_count")).collect()
    )
    combined = combined.join(play_counts, on=["user_id", "song_id"], how="left")

    combined.write_parquet(interim_path)
    logger.info("Saved interim dataset to %s", interim_path)
    logger.info(
        "Rows: train=%d, val=%d, test=%d",
        train_df.height,
        val_df.height,
        test_df.height,
    )

    dates_info.write_parquet(str(Path(interim_path).parent / "split_dates.parquet"))


if __name__ == "__main__":
    main()
