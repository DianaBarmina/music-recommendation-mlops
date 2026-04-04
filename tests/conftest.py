import pytest


@pytest.fixture
def sample_interactions_df():
    """Пример датасета взаимодействий"""
    pass


@pytest.fixture
def sample_interactions_with_duplicates():
    """Датасет с дубликатами"""
    pass


@pytest.fixture
def sample_user_mapping():
    """Маппинг user_id → индекс"""
    pass


@pytest.fixture
def sample_item_mapping():
    """Маппинг item_id → индекс"""
    pass


@pytest.fixture
def sample_sparse_matrix():
    """Простая разреженная матрица взаимодействий"""
    pass


@pytest.fixture
def perfect_recommendations():
    """Идеальные рекомендации"""
    pass


@pytest.fixture
def empty_recommendations():
    """Пустые рекомендации"""
    pass
