import os
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path):
    db_path = tmp_path / "test_db.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    from services.api.database import Base, engine
    from services.api.dependencies import artifacts

    with patch.object(artifacts, "load", autospec=True) as _:
        artifacts.model = object()
        artifacts.train_matrix = None
        artifacts.users_map = pl.DataFrame({"user_id": ["user_0"], "user_idx": [0]})
        artifacts.items_map = pl.DataFrame({"song_id": ["song_0"], "item_idx": [0]})
        artifacts.model_version = "test_v1"
        artifacts.is_ready = True

        Base.metadata.create_all(bind=engine)

        from services.api.main import app

        with (
            patch(
                "services.api.routers.recommendations.get_user_idx",
                return_value=0,
            ),
            patch(
                "services.api.routers.recommendations.get_recommendations",
                return_value=(["song_0"], [0.9]),
            ),
        ):
            with TestClient(app) as c:
                yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_recommendations_ok(client):
    r = client.get("/recommendations/user_0?n_items=1")
    assert r.status_code == 200
    data = r.json()
    assert data["user_id"] == "user_0"
    assert isinstance(data["recommendations"], list)
    assert isinstance(data["scores"], list)


def test_unknown_user_404(client):
    with patch("services.api.routers.recommendations.get_user_idx", return_value=None):
        r = client.get("/recommendations/unknown_user")
        assert r.status_code == 404
