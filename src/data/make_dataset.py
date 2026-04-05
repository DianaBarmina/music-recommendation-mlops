from datetime import timedelta
from pathlib import Path

import polars as pl

from src.utils.helpers import get_logger, load_params

logger = get_logger(__name__)


def load_raw_data(january_path: str, february_path: str) -> pl.LazyFrame:
    """
    Загружает январь и февраль, объединяет в один LazyFrame.
    Оставляет только колонки user_id, song_id, ts.
    """
    jan = pl.scan_parquet(january_path).select(["user_id", "song_id", "ts"])
    feb = pl.scan_parquet(february_path).select(["user_id", "song_id", "ts"])
    return pl.concat([jan, feb])


def filter_cold_users_and_items(
    train_pairs: pl.LazyFrame,
    min_user_items: int,
    min_item_users: int,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """
    Возвращает (good_users, good_items, filtered_train_pairs).
    good_users — пользователи с >= min_user_items уникальных треков в трейне.
    good_items — треки с >= min_item_users уникальных пользователей в трейне.
    """
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
    return good_users, good_items, filtered


def split_by_time_window(
    lf: pl.LazyFrame,
    good_users: pl.LazyFrame,
    good_items: pl.LazyFrame,
    train_pairs: pl.LazyFrame,
    test_window_days: int,
    val_window_days: int,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.DataFrame]:
    """
    Стратегия фиксированного окна:
    - Берём максимальную дату в данных
    - test = последние test_window_days дней
    - val  = предыдущие val_window_days дней перед тестом
    - train = всё остальное

    Возвращает (train_pairs_filtered, val_pairs, test_pairs, dates_info).
    val и test содержат только новые пары (anti join с трейном).
    val и test содержат только known users и items из трейна.
    """
    max_ts = lf.select(pl.col("ts").max()).collect().item()

    test_start = max_ts - timedelta(days=test_window_days)
    val_start = test_start - timedelta(days=val_window_days)

    logger.info(f"Max date: {max_ts}")
    logger.info(f"Train: до {val_start}")
    logger.info(f"Val:   {val_start} — {test_start}")
    logger.info(f"Test:  {test_start} — {max_ts}")

    # train_raw = lf.filter(pl.col("ts") < val_start)
    val_raw = lf.filter((pl.col("ts") >= val_start) & (pl.col("ts") < test_start))
    test_raw = lf.filter(pl.col("ts") >= test_start)

    # train_pairs уже отфильтрован по good_users/good_items
    val_pairs = (
        val_raw.select(["user_id", "song_id"])
        .unique()
        .join(good_users, on="user_id", how="inner")
        .join(good_items, on="song_id", how="inner")
        .join(train_pairs, on=["user_id", "song_id"], how="anti")
    )
    test_pairs = (
        test_raw.select(["user_id", "song_id"])
        .unique()
        .join(good_users, on="user_id", how="inner")
        .join(good_items, on="song_id", how="inner")
        .join(train_pairs, on=["user_id", "song_id"], how="anti")
    )

    dates_info = pl.DataFrame(
        {
            "val_start": [val_start],
            "test_start": [test_start],
            "max_ts": [max_ts],
        }
    )

    return train_pairs, val_pairs, test_pairs, dates_info


def main():
    params = load_params()

    january_path = params["data"]["january_path"]
    february_path = params["data"]["february_path"]
    interim_path = params["data"]["interim_path"]
    min_user_items = params["split"]["min_user_items"]
    min_item_users = params["split"]["min_item_users"]
    test_window_days = params["split"]["test_window_days"]
    val_window_days = params["split"]["val_window_days"]

    Path(interim_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw data...")
    lf = load_raw_data(january_path, february_path)

    total_window = test_window_days + val_window_days
    max_ts = lf.select(pl.col("ts").max()).collect().item()
    train_cutoff = max_ts - timedelta(days=total_window)

    train_raw_pairs = (
        lf.filter(pl.col("ts") < train_cutoff).select(["user_id", "song_id"]).unique()
    )

    logger.info("Filtering cold users and items...")
    good_users, good_items, train_pairs_filtered = filter_cold_users_and_items(
        train_raw_pairs, min_user_items, min_item_users
    )

    train_pairs_filtered_collected = train_pairs_filtered.collect()
    good_users_collected = good_users.collect()
    good_items_collected = good_items.collect()

    logger.info(
        f"After filtering: "
        f"{good_users_collected.height} users, "
        f"{good_items_collected.height} items"
    )

    train_final, val_pairs, test_pairs, dates_info = split_by_time_window(
        lf,
        good_users_collected.lazy(),
        good_items_collected.lazy(),
        train_pairs_filtered_collected.lazy(),
        test_window_days,
        val_window_days,
    )

    train_df = train_pairs_filtered_collected.with_columns(
        pl.lit("train").alias("split")
    )
    val_df = val_pairs.collect().with_columns(pl.lit("val").alias("split"))
    test_df = test_pairs.collect().with_columns(pl.lit("test").alias("split"))

    combined = pl.concat([train_df, val_df, test_df], how="diagonal")

    # Добавляем play_count из оригинальных данных
    play_counts = (
        lf.group_by(["user_id", "song_id"]).agg(pl.len().alias("play_count")).collect()
    )

    combined = combined.join(play_counts, on=["user_id", "song_id"], how="left")

    combined.write_parquet(interim_path)
    logger.info(f"Saved interim dataset to {interim_path}")
    logger.info(
        f"Rows: train={train_df.height}, val={val_df.height}, test={test_df.height}"
    )

    dates_info.write_parquet(str(Path(interim_path).parent / "split_dates.parquet"))


if __name__ == "__main__":
    main()
