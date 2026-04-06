import pickle

import numpy as np
import polars as pl
import pytest
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


def make_train_matrix(n_users: int = 20, n_items: int = 50, density: float = 0.1):
    rng = np.random.default_rng(42)
    data_size = int(n_users * n_items * density)
    rows = rng.integers(0, n_users, size=data_size)
    cols = rng.integers(0, n_items, size=data_size)
    data = rng.uniform(0.1, 5.0, size=data_size).astype(np.float32)
    matrix = csr_matrix(
        (data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32
    )
    return matrix


def make_trained_model(train_matrix: csr_matrix) -> AlternatingLeastSquares:
    model = AlternatingLeastSquares(
        factors=8,
        iterations=3,
        regularization=0.01,
        random_state=42,
        use_gpu=False,
        num_threads=1,
    )

    model.fit(train_matrix, show_progress=False)
    return model


@pytest.fixture(scope="module")
def trained_artifacts():
    n_users, n_items = 20, 50
    train_matrix = make_train_matrix(n_users, n_items)
    model = make_trained_model(train_matrix)
    users_map = pl.DataFrame(
        {
            "user_id": [f"user_{i}" for i in range(n_users)],
            "user_idx": list(range(n_users)),
        }
    )
    items_map = pl.DataFrame(
        {
            "song_id": [f"song_{i}" for i in range(n_items)],
            "item_idx": list(range(n_items)),
        }
    )
    return model, train_matrix, users_map, items_map


class TestALSTraining:
    def test_model_trains_without_error(self):
        matrix = make_train_matrix()
        model = make_trained_model(matrix)
        assert model is not None

    def test_model_has_user_factors(self, trained_artifacts):
        model, train_matrix, _, _ = trained_artifacts
        assert model.user_factors is not None
        assert model.user_factors.shape[0] == train_matrix.shape[0]

    def test_model_has_item_factors(self, trained_artifacts):
        model, train_matrix, _, _ = trained_artifacts
        assert model.item_factors is not None
        assert model.item_factors.shape[0] == train_matrix.shape[1]

    def test_factors_dimension_correct(self, trained_artifacts):
        model, _, _, _ = trained_artifacts
        assert model.user_factors.shape[1] == 8
        assert model.item_factors.shape[1] == 8

    def test_model_serializable(self, trained_artifacts, tmp_path):
        model, _, _, _ = trained_artifacts
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)
        assert loaded is not None
        assert loaded.user_factors.shape == model.user_factors.shape


class TestRecommendations:
    def test_recommendations_correct_count(self, trained_artifacts):
        from src.models.predict_model import get_recommendations

        model, train_matrix, users_map, items_map = trained_artifacts
        songs, scores = get_recommendations(
            user_id="user_0",
            n_items=10,
            model=model,
            train_matrix=train_matrix,
            users_map=users_map,
            items_map=items_map,
        )
        assert len(songs) == 10
        assert len(scores) == 10

    def test_recommendations_are_strings(self, trained_artifacts):
        from src.models.predict_model import get_recommendations

        model, train_matrix, users_map, items_map = trained_artifacts
        songs, _ = get_recommendations(
            user_id="user_0",
            n_items=5,
            model=model,
            train_matrix=train_matrix,
            users_map=users_map,
            items_map=items_map,
        )
        assert all(isinstance(s, str) for s in songs)

    def test_scores_are_floats(self, trained_artifacts):
        from src.models.predict_model import get_recommendations

        model, train_matrix, users_map, items_map = trained_artifacts
        _, scores = get_recommendations(
            user_id="user_0",
            n_items=5,
            model=model,
            train_matrix=train_matrix,
            users_map=users_map,
            items_map=items_map,
        )
        assert all(isinstance(s, float) for s in scores)

    def test_scores_descending(self, trained_artifacts):
        from src.models.predict_model import get_recommendations

        model, train_matrix, users_map, items_map = trained_artifacts
        _, scores = get_recommendations(
            user_id="user_0",
            n_items=10,
            model=model,
            train_matrix=train_matrix,
            users_map=users_map,
            items_map=items_map,
        )
        assert scores == sorted(scores, reverse=True)

    def test_unknown_user_returns_empty(self, trained_artifacts):
        from src.models.predict_model import get_recommendations

        model, train_matrix, users_map, items_map = trained_artifacts
        songs, scores = get_recommendations(
            user_id="nonexistent_user_xyz",
            n_items=10,
            model=model,
            train_matrix=train_matrix,
            users_map=users_map,
            items_map=items_map,
        )
        assert songs == []
        assert scores == []

    def test_no_already_seen_items(self, trained_artifacts):
        from src.models.predict_model import get_recommendations

        model, train_matrix, users_map, items_map = trained_artifacts

        user_idx = 0
        seen_indices = set(train_matrix[user_idx].indices.tolist())
        idx_to_song = dict(
            zip(items_map["item_idx"].to_list(), items_map["song_id"].to_list())
        )
        seen_songs = {idx_to_song[i] for i in seen_indices if i in idx_to_song}

        songs, _ = get_recommendations(
            user_id="user_0",
            n_items=10,
            model=model,
            train_matrix=train_matrix,
            users_map=users_map,
            items_map=items_map,
        )
        for song in songs:
            assert song not in seen_songs

    def test_n_items_one(self, trained_artifacts):
        from src.models.predict_model import get_recommendations

        model, train_matrix, users_map, items_map = trained_artifacts
        songs, scores = get_recommendations(
            user_id="user_0",
            n_items=1,
            model=model,
            train_matrix=train_matrix,
            users_map=users_map,
            items_map=items_map,
        )
        assert len(songs) == 1
        assert len(scores) == 1


class TestGetUserIdx:
    def test_known_user_returns_int(self, trained_artifacts):
        from src.models.predict_model import get_user_idx

        _, _, users_map, _ = trained_artifacts
        result = get_user_idx("user_0", users_map)
        assert isinstance(result, (int, np.integer))

    def test_unknown_user_returns_none(self, trained_artifacts):
        from src.models.predict_model import get_user_idx

        _, _, users_map, _ = trained_artifacts
        result = get_user_idx("nobody_xyz", users_map)
        assert result is None

    def test_idx_in_valid_range(self, trained_artifacts):
        from src.models.predict_model import get_user_idx

        _, train_matrix, users_map, _ = trained_artifacts
        idx = get_user_idx("user_5", users_map)
        assert 0 <= idx < train_matrix.shape[0]
