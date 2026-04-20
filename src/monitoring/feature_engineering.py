import pandas as pd
import polars as pl

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def aggregate_daily_features(df: pl.DataFrame) -> pd.DataFrame:
    """
    Из сырых интеракций за один день строит агрегированные фичи
    на уровне пользователя. Это то, что будет сравнивать Evidently.

    Колонки входного df: user_id, song_id, play_count (или ts).
    Возвращает DataFrame с одной строкой на пользователя.
    """
    user_stats = (
        df.group_by("user_id")
        .agg(
            [
                pl.len().alias("n_interactions"),
                pl.col("song_id").n_unique().alias("n_unique_songs"),
                pl.col("play_count").mean().alias("avg_play_count"),
                pl.col("play_count").median().alias("median_play_count"),
                pl.col("play_count").std().alias("std_play_count"),
                pl.col("play_count").max().alias("max_play_count"),
                pl.col("play_count").sum().alias("total_play_count"),
            ]
        )
        .with_columns(
            [
                # отношение уникальных треков к общему числу прослушиваний
                (pl.col("n_unique_songs") / pl.col("n_interactions")).alias(
                    "diversity_ratio"
                ),
            ]
        )
    )
    return user_stats.to_pandas()


def aggregate_reference_features(reference_path: str) -> pd.DataFrame:
    """
    Строит агрегированные фичи для reference датасета (янв+фев)
    Делает то же самое что aggregate_daily_features но для всего reference
    """
    df = pl.read_parquet(reference_path)
    if "play_count" not in df.columns:
        df = df.with_columns(pl.lit(1).alias("play_count"))
    return aggregate_daily_features(df)


def compute_dataset_stats(df: pl.DataFrame) -> dict:
    """
    Скалярные статистики датасета за день
    Сохраняются в таблицу daily_stats
    """
    if "play_count" not in df.columns:
        df = df.with_columns(pl.lit(1).alias("play_count"))

    return {
        "n_interactions": df.height,
        "n_unique_users": df["user_id"].n_unique(),
        "n_unique_items": df["song_id"].n_unique(),
        "avg_play_count": float(df["play_count"].mean() or 0),
        "median_play_count": float(df["play_count"].median() or 0),
        "std_play_count": float(df["play_count"].std() or 0),
    }
