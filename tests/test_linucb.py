"""Unit tests for the explicit-update LinUCB policy."""

import math
import unittest

from src.simulation.linucb import DEFAULT_N_FEATURES, LinUCB


class LinUCBTest(unittest.TestCase):
    def test_initialization_uses_identity_zero_and_zero_theta(self) -> None:
        model = LinUCB(alpha=0.2)
        self.assertEqual(model.n_features, 15)
        self.assertEqual(model.alpha, 0.2)
        self.assertEqual(len(model.A), 15)
        self.assertEqual(model.A, tuple(
            tuple(1.0 if i == j else 0.0 for j in range(15))
            for i in range(15)
        ))
        self.assertEqual(model.b, (0.0,) * 15)
        self.assertEqual(model.theta, (0.0,) * 15)

    def test_default_dimension_and_configurable_dimension(self) -> None:
        self.assertEqual(DEFAULT_N_FEATURES, 15)
        model = LinUCB(n_features=2, alpha=0.0)
        self.assertEqual(model.A, ((1.0, 0.0), (0.0, 1.0)))
        with self.assertRaisesRegex(ValueError, "exactly 2 features"):
            model.score((1.0, 2.0, 3.0))
        for invalid_dimension in (0, -2, True, 2.5):
            with self.subTest(dimension=invalid_dimension):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    LinUCB(n_features=invalid_dimension, alpha=0.0)

    def test_alpha_is_explicit_nonnegative_numeric_and_finite(self) -> None:
        self.assertEqual(LinUCB(alpha=0).alpha, 0.0)
        self.assertEqual(LinUCB(alpha=1.75).alpha, 1.75)
        for invalid_alpha in (-0.1, float("nan"), float("inf"), -float("inf"), "0.2", True):
            with self.subTest(alpha=invalid_alpha):
                with self.assertRaisesRegex(ValueError, "alpha"):
                    LinUCB(alpha=invalid_alpha)

    def test_initial_score_is_alpha_times_context_norm(self) -> None:
        alpha = 0.3
        context = tuple(float(value) for value in range(1, 16))
        model = LinUCB(alpha=alpha)
        self.assertAlmostEqual(model.score(context), alpha * math.sqrt(1240))

    def test_mean_and_exploration_components_sum_to_score(self) -> None:
        model = LinUCB(n_features=2, alpha=0.3)
        model.update((1.0, 0.0), 2.0)
        result = model.score_components((1.0, 2.0))
        self.assertAlmostEqual(result.score, result.mean + result.exploration)
        self.assertAlmostEqual(result.mean, 1.0)
        self.assertGreater(result.exploration, 0.0)

    def test_top1_selects_the_highest_scoring_candidate(self) -> None:
        model = LinUCB(n_features=2, alpha=0.0)
        model.update((1.0, 0.0), 2.0)
        contexts = ((1.0, 0.0), (0.0, 1.0), (2.0, 0.0))
        selected = model.select_top1(contexts, candidate_ids=("a", "b", "c"))
        self.assertEqual(selected.index, 2)
        self.assertEqual(selected.candidate_id, "c")
        self.assertAlmostEqual(selected.score, 2.0)

    def test_top_k_is_ordered_and_limited_to_pool_size(self) -> None:
        model = LinUCB(n_features=2, alpha=0.0)
        model.update((1.0, 0.0), 2.0)
        contexts = ((0.0, 1.0), (1.0, 0.0), (2.0, 0.0))
        ranked = model.top_k(contexts, 2)
        self.assertEqual(tuple(item.index for item in ranked), (2, 1))
        self.assertEqual(len(model.top_k(contexts, 20)), 3)
        self.assertEqual(model.top_k((), 3), ())
        with self.assertRaisesRegex(ValueError, "positive integer"):
            model.top_k(contexts, 0)

    def test_top_k_does_not_update_model(self) -> None:
        model = LinUCB(n_features=2, alpha=0.2)
        before = (model.A, model.b, model.theta)
        model.top_k(((1.0, 0.0), (0.0, 1.0)), 2)
        self.assertEqual((model.A, model.b, model.theta), before)

    def test_top1_selection_does_not_update_model(self) -> None:
        model = LinUCB(n_features=2, alpha=0.2)
        before = (model.A, model.b, model.theta)
        model.select_top1(((1.0, 0.0), (0.0, 1.0)))
        self.assertEqual((model.A, model.b, model.theta), before)

    def test_update_applies_exact_rank_one_matrix_and_response_updates(self) -> None:
        model = LinUCB(n_features=2, alpha=0.4)
        model.update((1.0, 2.0), 3.0)
        self.assertEqual(model.A, ((2.0, 2.0), (2.0, 5.0)))
        self.assertEqual(model.b, (3.0, 6.0))
        self.assertAlmostEqual(model.theta[0], 0.5)
        self.assertAlmostEqual(model.theta[1], 1.0)

    def test_zero_reward_updates_matrix_but_not_response(self) -> None:
        model = LinUCB(n_features=2, alpha=0.1)
        model.update((2.0, -1.0), 0.0)
        self.assertEqual(model.A, ((5.0, -2.0), (-2.0, 2.0)))
        self.assertEqual(model.b, (0.0, 0.0))
        self.assertEqual(model.theta, (0.0, 0.0))

    def test_negative_reward_is_allowed(self) -> None:
        model = LinUCB(n_features=2, alpha=0.1)
        model.update((1.0, 2.0), -2.0)
        self.assertEqual(model.b, (-2.0, -4.0))
        self.assertAlmostEqual(model.theta[0], -1.0 / 3.0)
        self.assertAlmostEqual(model.theta[1], -2.0 / 3.0)

    def test_scoring_is_deterministic_and_ties_keep_input_order(self) -> None:
        model = LinUCB(n_features=2, alpha=1.0)
        contexts = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0))
        first = model.top_k(contexts, 3)
        second = model.top_k(contexts, 3)
        self.assertEqual(first, second)
        self.assertEqual(tuple(item.index for item in first), (0, 1, 2))

    def test_invalid_context_dimension_nan_and_infinity_are_rejected(self) -> None:
        model = LinUCB(n_features=2, alpha=0.1)
        for context in (
            (1.0,),
            (1.0, 2.0, 3.0),
            (float("nan"), 1.0),
            (1.0, float("inf")),
            ("1", 2.0),
        ):
            with self.subTest(context=context):
                with self.assertRaises(ValueError):
                    model.score(context)

    def test_invalid_reward_nan_or_infinity_is_rejected_without_mutation(self) -> None:
        model = LinUCB(n_features=2, alpha=0.1)
        before = (model.A, model.b)
        for reward in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(reward=reward):
                with self.assertRaisesRegex(ValueError, "reward"):
                    model.update((1.0, 2.0), reward)
        self.assertEqual((model.A, model.b), before)

    def test_numerical_reference_score_after_update(self) -> None:
        model = LinUCB(n_features=2, alpha=0.3)
        model.update((1.0, 2.0), 3.0)
        result = model.score_components((1.0, 2.0))
        self.assertAlmostEqual(result.mean, 2.5)
        self.assertAlmostEqual(result.exploration, 0.3 * math.sqrt(5.0 / 6.0))
        self.assertAlmostEqual(result.score, 2.5 + 0.3 * math.sqrt(5.0 / 6.0))

    def test_matrix_remains_symmetric_after_updates(self) -> None:
        model = LinUCB(n_features=3, alpha=0.1)
        model.update((1.0, -2.0, 0.5), 1.0)
        model.update((0.2, 3.0, -1.0), -0.5)
        for row_index in range(3):
            for column_index in range(3):
                self.assertEqual(
                    model.A[row_index][column_index],
                    model.A[column_index][row_index],
                )

    def test_multiple_updates_match_manual_accumulation(self) -> None:
        model = LinUCB(n_features=2, alpha=0.0)
        model.update((1.0, 0.0), 2.0)
        model.update((0.0, 2.0), -1.0)
        self.assertEqual(model.A, ((2.0, 0.0), (0.0, 5.0)))
        self.assertEqual(model.b, (2.0, -2.0))
        self.assertAlmostEqual(model.theta[0], 1.0)
        self.assertAlmostEqual(model.theta[1], -0.4)

    def test_batch_ids_must_match_batch_size_and_empty_top1_fails(self) -> None:
        model = LinUCB(n_features=2, alpha=0.0)
        with self.assertRaisesRegex(ValueError, "length must match"):
            model.score_batch(((1.0, 0.0),), candidate_ids=("a", "b"))
        with self.assertRaisesRegex(ValueError, "empty candidate set"):
            model.select_top1(())


if __name__ == "__main__":
    unittest.main()
