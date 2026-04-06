import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from src.data.build_features import build_index_maps, encode_pairs, to_csr


def make_sample_df(
    n_users: int = 5, n_items: int = 10, n_rows: int = 30
) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            "user_id": [f"user_{rng.integers(0, n_users)}" for _ in range(n_rows)],
            "song_id": [f"song_{rng.integers(0, n_items)}" for _ in range(n_rows)],
            "play_count": rng.integers(1, 10, size=n_rows).tolist(),
            "split": ["train"] * n_rows,
        }
    )


class TestBuildIndexMaps:
    def test_user_mapping_is_unique(self):
        df = make_sample_df()
        users_map, _ = build_index_maps(df)
        assert users_map["user_idx"].n_unique() == users_map.height

    def test_item_mapping_is_unique(self):
        df = make_sample_df()
        _, items_map = build_index_maps(df)
        assert items_map["item_idx"].n_unique() == items_map.height

    def test_user_idx_starts_from_zero(self):
        df = make_sample_df()
        users_map, _ = build_index_maps(df)
        assert users_map["user_idx"].min() == 0

    def test_item_idx_starts_from_zero(self):
        df = make_sample_df()
        _, items_map = build_index_maps(df)
        assert items_map["item_idx"].min() == 0

    def test_user_idx_is_contiguous(self):
        df = make_sample_df()
        users_map, _ = build_index_maps(df)
        expected = set(range(users_map.height))
        actual = set(users_map["user_idx"].to_list())
        assert actual == expected

    def test_all_train_users_in_map(self):
        df = make_sample_df()
        users_map, _ = build_index_maps(df)
        train_users = set(df["user_id"].unique().to_list())
        map_users = set(users_map["user_id"].to_list())
        assert train_users == map_users

    def test_mapping_columns_exist(self):
        df = make_sample_df()
        users_map, items_map = build_index_maps(df)
        assert "user_id" in users_map.columns
        assert "user_idx" in users_map.columns
        assert "song_id" in items_map.columns
        assert "item_idx" in items_map.columns


class TestEncodePairs:
    def test_encoded_has_idx_columns(self):
        df = make_sample_df()
        users_map, items_map = build_index_maps(df)
        encoded = encode_pairs(df, users_map, items_map)
        assert "user_idx" in encoded.columns
        assert "item_idx" in encoded.columns

    def test_unknown_users_dropped(self):
        df = make_sample_df()
        users_map, items_map = build_index_maps(df)

        extra = pl.DataFrame(
            {
                "user_id": ["unknown_user_999"],
                "song_id": ["song_0"],
                "play_count": [1],
                "split": ["val"],
            }
        )
        encoded = encode_pairs(extra, users_map, items_map)
        assert encoded.is_empty()

    def test_no_rows_lost_for_known_pairs(self):
        df = make_sample_df().unique(subset=["user_id", "song_id"])
        users_map, items_map = build_index_maps(df)
        encoded = encode_pairs(df, users_map, items_map)
        assert encoded.height == df.height


class TestToCSR:
    def _make_encoded(self):
        df = make_sample_df()
        users_map, items_map = build_index_maps(df)
        encoded = encode_pairs(df, users_map, items_map).with_columns(
            pl.col("play_count").cast(pl.Float32).alias("value")
        )
        return encoded, users_map.height, items_map.height

    def test_matrix_shape_matches_mappings(self):
        encoded, n_users, n_items = self._make_encoded()
        matrix = to_csr(encoded, n_users, n_items)
        assert matrix.shape == (n_users, n_items)

    def test_matrix_is_sparse(self):
        encoded, n_users, n_items = self._make_encoded()
        matrix = to_csr(encoded, n_users, n_items)
        assert isinstance(matrix, csr_matrix)

    def test_matrix_dtype_is_float32(self):
        encoded, n_users, n_items = self._make_encoded()
        matrix = to_csr(encoded, n_users, n_items)
        assert matrix.dtype == np.float32

    def test_matrix_nnz_positive(self):
        encoded, n_users, n_items = self._make_encoded()
        matrix = to_csr(encoded, n_users, n_items)
        assert matrix.nnz > 0

    def test_all_values_positive(self):
        encoded, n_users, n_items = self._make_encoded()
        matrix = to_csr(encoded, n_users, n_items)
        assert (matrix.data > 0).all()


class TestColdStartFiltering:
    def _filter(self, df: pl.DataFrame, min_user: int, min_item: int):
        pairs = df.select(["user_id", "song_id"]).unique()

        good_users = (
            pairs.group_by("user_id")
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") >= min_user)
            .select("user_id")
        )
        good_items = (
            pairs.group_by("song_id")
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") >= min_item)
            .select("song_id")
        )
        filtered = pairs.join(good_users, on="user_id", how="inner").join(
            good_items, on="song_id", how="inner"
        )
        return filtered, good_users, good_items

    def test_all_users_meet_threshold(self):
        df = pl.DataFrame(
            {
                "user_id": ["u1"] * 5 + ["u2"] * 5,
                "song_id": [f"s{i}" for i in range(10)],
            }
        )
        filtered, good_users, _ = self._filter(df, min_user=3, min_item=1)
        assert good_users.height == 2

    def test_cold_user_removed(self):
        df = pl.DataFrame(
            {
                "user_id": ["u1"] * 5 + ["cold_user"],
                "song_id": [f"s{i}" for i in range(5)] + ["s0"],
            }
        )
        filtered, good_users, _ = self._filter(df, min_user=3, min_item=1)
        user_ids = good_users["user_id"].to_list()
        assert "cold_user" not in user_ids
        assert "u1" in user_ids

    def test_cold_item_removed(self):
        df = pl.DataFrame(
            {
                "user_id": [f"u{i}" for i in range(5)] + ["u0"],
                "song_id": ["popular"] * 5 + ["cold_song"],
            }
        )
        filtered, _, good_items = self._filter(df, min_user=1, min_item=3)
        item_ids = good_items["song_id"].to_list()
        assert "cold_song" not in item_ids
        assert "popular" in item_ids

    def test_empty_after_strict_filter(self):
        df = pl.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "song_id": ["s1", "s2"],
            }
        )
        filtered, good_users, good_items = self._filter(df, min_user=100, min_item=100)
        assert good_users.is_empty()
        assert good_items.is_empty()
