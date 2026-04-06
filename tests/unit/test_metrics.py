import numpy as np

from src.models.metrics import (
    hit_rate_at_k,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestNdcgAtK:
    def test_perfect_recommendations_give_one(self):
        recs = np.array([0, 1, 2, 3, 4])
        true_items = np.array([0, 1, 2, 3, 4])
        assert abs(ndcg_at_k(recs, true_items, k=5) - 1.0) < 1e-6

    def test_no_hits_give_zero(self):
        recs = np.array([10, 11, 12])
        true_items = np.array([0, 1, 2])
        assert ndcg_at_k(recs, true_items, k=3) == 0.0

    def test_partial_hits_between_zero_and_one(self):
        recs = np.array([0, 10, 2])
        true_items = np.array([0, 1, 2])
        result = ndcg_at_k(recs, true_items, k=3)
        assert 0.0 < result < 1.0

    def test_earlier_hit_gives_higher_score(self):
        true_items = np.array([5])
        recs_early = np.array([5, 10, 11])
        recs_late = np.array([10, 11, 5])
        assert ndcg_at_k(recs_early, true_items, k=3) > ndcg_at_k(
            recs_late, true_items, k=3
        )

    def test_empty_true_items_give_zero(self):
        recs = np.array([0, 1, 2])
        true_items = np.array([], dtype=np.int64)
        assert ndcg_at_k(recs, true_items, k=3) == 0.0

    def test_k_truncates_recs(self):
        recs = np.array([10, 11, 12, 5])
        true_items = np.array([5])
        assert ndcg_at_k(recs, true_items, k=3) == 0.0
        assert ndcg_at_k(recs, true_items, k=4) > 0.0

    def test_known_single_hit_at_position_one(self):
        recs = np.array([5])
        true_items = np.array([5])
        assert abs(ndcg_at_k(recs, true_items, k=1) - 1.0) < 1e-6


class TestPrecisionAtK:
    def test_all_hits_give_one(self):
        recs = np.array([0, 1, 2])
        true_items = np.array([0, 1, 2, 3])
        assert precision_at_k(recs, true_items, k=3) == 1.0

    def test_no_hits_give_zero(self):
        recs = np.array([10, 11, 12])
        true_items = np.array([0, 1, 2])
        assert precision_at_k(recs, true_items, k=3) == 0.0

    def test_half_hits(self):
        recs = np.array([0, 10, 2, 11])
        true_items = np.array([0, 2])
        assert abs(precision_at_k(recs, true_items, k=4) - 0.5) < 1e-6

    def test_k_truncates(self):
        recs = np.array([10, 11, 0, 1])
        true_items = np.array([0, 1])
        assert precision_at_k(recs, true_items, k=2) == 0.0

    def test_k_larger_than_recs(self):
        recs = np.array([0, 1])
        true_items = np.array([0, 1])
        result = precision_at_k(recs, true_items, k=10)
        assert abs(result - 1.0) < 1e-6


class TestRecallAtK:
    def test_all_relevant_found(self):
        recs = np.array([0, 1, 2])
        true_items = np.array([0, 1])
        assert recall_at_k(recs, true_items, k=3) == 1.0

    def test_none_found(self):
        recs = np.array([10, 11])
        true_items = np.array([0, 1])
        assert recall_at_k(recs, true_items, k=2) == 0.0

    def test_partial_recall(self):
        recs = np.array([0, 10])
        true_items = np.array([0, 1, 2, 3])
        assert abs(recall_at_k(recs, true_items, k=2) - 0.25) < 1e-6

    def test_empty_true_items_give_zero(self):
        recs = np.array([0, 1, 2])
        true_items = np.array([], dtype=np.int64)
        assert recall_at_k(recs, true_items, k=3) == 0.0

    def test_recall_can_exceed_precision(self):
        recs = np.array([0])
        true_items = np.array([0, 1, 2, 3, 4])
        p = precision_at_k(recs, true_items, k=1)
        r = recall_at_k(recs, true_items, k=1)
        assert p == 1.0
        assert r == 0.2


class TestHitRateAtK:
    def test_one_hit_gives_one(self):
        recs = np.array([99, 1, 88])
        true_items = np.array([1])
        assert hit_rate_at_k(recs, true_items, k=3) == 1.0

    def test_no_hits_give_zero(self):
        recs = np.array([10, 11, 12])
        true_items = np.array([0, 1])
        assert hit_rate_at_k(recs, true_items, k=3) == 0.0

    def test_hit_at_boundary_position_k(self):
        recs = np.array([10, 11, 5])
        true_items = np.array([5])
        assert hit_rate_at_k(recs, true_items, k=3) == 1.0

    def test_hit_beyond_k_not_counted(self):
        recs = np.array([10, 11, 5])
        true_items = np.array([5])
        assert hit_rate_at_k(recs, true_items, k=2) == 0.0

    def test_binary_output(self):
        recs = np.array([0, 1, 2, 3, 4])
        true_items = np.array([0, 1])
        result = hit_rate_at_k(recs, true_items, k=5)
        assert result in (0.0, 1.0)


class TestMrrAtK:
    def test_first_position_gives_one(self):
        recs = np.array([5, 10, 11])
        true_items = np.array([5])
        assert abs(mrr_at_k(recs, true_items, k=3) - 1.0) < 1e-6

    def test_second_position_gives_half(self):
        recs = np.array([10, 5, 11])
        true_items = np.array([5])
        assert abs(mrr_at_k(recs, true_items, k=3) - 0.5) < 1e-6

    def test_third_position_gives_one_third(self):
        recs = np.array([10, 11, 5])
        true_items = np.array([5])
        assert abs(mrr_at_k(recs, true_items, k=3) - 1.0 / 3.0) < 1e-6

    def test_no_hit_gives_zero(self):
        recs = np.array([10, 11, 12])
        true_items = np.array([5])
        assert mrr_at_k(recs, true_items, k=3) == 0.0

    def test_uses_first_hit_only(self):
        recs = np.array([10, 5, 6])
        true_items = np.array([5, 6])
        assert abs(mrr_at_k(recs, true_items, k=3) - 0.5) < 1e-6

    def test_hit_beyond_k_not_counted(self):
        recs = np.array([10, 11, 5])
        true_items = np.array([5])
        assert mrr_at_k(recs, true_items, k=2) == 0.0


class TestMapAtK:
    def test_perfect_gives_one(self):
        recs = np.array([0, 1, 2])
        true_items = np.array([0, 1, 2])
        assert abs(map_at_k(recs, true_items, k=3) - 1.0) < 1e-6

    def test_no_hits_gives_zero(self):
        recs = np.array([10, 11, 12])
        true_items = np.array([0, 1, 2])
        assert map_at_k(recs, true_items, k=3) == 0.0

    def test_known_value(self):
        recs = np.array([0, 10, 1])
        true_items = np.array([0, 1])
        expected = (1.0 + 2.0 / 3.0) / 2.0
        assert abs(map_at_k(recs, true_items, k=3) - expected) < 1e-6

    def test_single_hit_at_position_two(self):
        recs = np.array([10, 5, 11])
        true_items = np.array([5])
        assert abs(map_at_k(recs, true_items, k=3) - 0.5) < 1e-6

    def test_order_matters(self):
        true_items = np.array([0, 1])
        recs_good = np.array([0, 1, 10])
        recs_bad = np.array([10, 0, 1])
        assert map_at_k(recs_good, true_items, k=3) > map_at_k(
            recs_bad, true_items, k=3
        )

    def test_hit_beyond_k_not_counted(self):
        recs = np.array([10, 11, 5])
        true_items = np.array([5])
        assert map_at_k(recs, true_items, k=2) == 0.0
        assert map_at_k(recs, true_items, k=3) > 0.0
