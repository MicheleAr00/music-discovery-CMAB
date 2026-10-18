"""Tests for warm-up-only context feature standardization."""

from datetime import datetime, timedelta
import math
import unittest

from src.data.state import PlayEvent, TemporalState
from src.simulation.candidate_generator import CandidateGenerator
from src.simulation.context import ContextBuilder
from src.simulation.scaler import FeatureScaler


def row(offset: float, *, track_novelty: float = 0, artist_novelty: float = 1):
    """Build an easily inspected raw 15-feature context."""
    return tuple(offset + index for index in range(13)) + (
        float(track_novelty),
        float(artist_novelty),
    )


class FeatureScalerTest(unittest.TestCase):
    def test_single_vector_and_batch_keep_their_15_feature_shapes(self) -> None:
        scaler = FeatureScaler().fit((row(0), row(2)))
        transformed_single = scaler.transform(row(1))
        transformed_batch = scaler.transform((row(0), row(2)))
        self.assertEqual(len(transformed_single), 15)
        self.assertEqual(len(transformed_batch), 2)
        self.assertTrue(all(len(vector) == 15 for vector in transformed_batch))
        self.assertTrue(all(type(value) is float for value in transformed_single))

    def test_fit_uses_population_mean_and_standard_deviation_ddof_zero(self) -> None:
        contexts = (row(0), row(2), row(4))
        scaler = FeatureScaler().fit(contexts)
        expected_scale = math.sqrt(8 / 3)
        for index in range(13):
            self.assertAlmostEqual(scaler.mean_[index], index + 2)
            self.assertAlmostEqual(scaler.scale_[index], expected_scale)
        self.assertEqual(scaler.mean_[13:], (0.0, 0.0))
        self.assertEqual(scaler.scale_[13:], (1.0, 1.0))

    def test_standardization_matches_formula_for_numeric_features(self) -> None:
        contexts = (row(0), row(2), row(4))
        scaler = FeatureScaler().fit(contexts)
        transformed = scaler.transform(row(2))
        for index in range(13):
            expected = (row(2)[index] - scaler.mean_[index]) / scaler.scale_[index]
            self.assertAlmostEqual(transformed[index], expected)
            self.assertEqual(transformed[index], 0.0)

    def test_binary_novelty_columns_remain_unstandardized_zero_or_one(self) -> None:
        scaler = FeatureScaler().fit((row(0, track_novelty=0, artist_novelty=1),
                                      row(2, track_novelty=1, artist_novelty=0)))
        outputs = scaler.transform(
            (row(1, track_novelty=0, artist_novelty=1),
             row(1, track_novelty=1, artist_novelty=0))
        )
        self.assertEqual(tuple(vector[13:] for vector in outputs), ((0.0, 1.0), (1.0, 0.0)))

    def test_zero_variance_uses_scale_one_and_is_deterministic(self) -> None:
        contexts = [list(row(0)), list(row(2)), list(row(4))]
        for context in contexts:
            context[5] = 7.0
        scaler = FeatureScaler().fit(contexts)
        self.assertEqual(scaler.mean_[5], 7.0)
        self.assertEqual(scaler.scale_[5], 1.0)
        first = scaler.transform(contexts[0])
        second = scaler.transform(contexts[0])
        self.assertEqual(first[5], 0.0)
        self.assertEqual(first, second)

    def test_transform_before_fit_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "fitted on warm-up"):
            FeatureScaler().transform(row(0))

    def test_fit_is_one_shot_and_transform_does_not_change_parameters(self) -> None:
        scaler = FeatureScaler().fit((row(0), row(2)))
        saved_mean, saved_scale = scaler.mean_, scaler.scale_
        scaler.transform(row(100))  # validation-like observation
        self.assertEqual(scaler.mean_, saved_mean)
        self.assertEqual(scaler.scale_, saved_scale)
        with self.assertRaisesRegex(RuntimeError, "already fitted"):
            scaler.fit((row(1000), row(2000)))
        self.assertEqual(scaler.mean_, saved_mean)
        self.assertEqual(scaler.scale_, saved_scale)

    def test_validation_batch_cannot_leak_into_fitted_parameters(self) -> None:
        scaler = FeatureScaler().fit((row(0), row(2)))
        saved_mean, saved_scale = scaler.mean_, scaler.scale_
        scaler.transform((row(10_000), row(-10_000)))
        self.assertEqual(scaler.mean_, saved_mean)
        self.assertEqual(scaler.scale_, saved_scale)

    def test_test_batch_cannot_leak_into_fitted_parameters(self) -> None:
        scaler = FeatureScaler().fit((row(0), row(2)))
        saved_mean, saved_scale = scaler.mean_, scaler.scale_
        scaler.transform((row(1e100), row(-1e100)))
        self.assertEqual(scaler.mean_, saved_mean)
        self.assertEqual(scaler.scale_, saved_scale)

    def test_same_warmup_is_deterministic(self) -> None:
        contexts = (row(0), row(2), row(4))
        first = FeatureScaler().fit(contexts)
        second = FeatureScaler().fit(contexts)
        self.assertEqual(first.mean_, second.mean_)
        self.assertEqual(first.scale_, second.scale_)
        self.assertEqual(first.transform(row(3)), second.transform(row(3)))

    def test_invalid_feature_counts_raise_clear_errors(self) -> None:
        scaler = FeatureScaler().fit(row(0))
        with self.assertRaisesRegex(ValueError, "exactly 15"):
            scaler.transform(tuple(range(14)))
        with self.assertRaisesRegex(ValueError, "exactly 15"):
            scaler.transform((row(0), tuple(range(14))))
        with self.assertRaisesRegex(ValueError, "exactly 15"):
            FeatureScaler().fit(tuple(range(16)))

    def test_non_binary_novelty_inputs_are_rejected(self) -> None:
        invalid = list(row(0))
        invalid[14] = 0.5
        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            FeatureScaler().fit(invalid)

    def test_fit_transform_matches_fit_then_transform(self) -> None:
        contexts = (row(0), row(2), row(4))
        direct_scaler = FeatureScaler()
        direct = direct_scaler.fit_transform(contexts)

        staged_scaler = FeatureScaler().fit(contexts)
        staged = staged_scaler.transform(contexts)
        self.assertEqual(direct, staged)
        self.assertEqual(direct_scaler.mean_, staged_scaler.mean_)
        self.assertEqual(direct_scaler.scale_, staged_scaler.scale_)

    def test_context_builder_candidate_generator_scaler_integration(self) -> None:
        base = datetime(2005, 10, 30, 10, 0)
        events = [
            PlayEvent.from_raw("U1", base, "T1", "A1"),
            PlayEvent.from_raw("U2", base + timedelta(seconds=1), "T2", "A2"),
        ]
        state = TemporalState(events)
        generator = CandidateGenerator(state)
        decision_time = base + timedelta(seconds=2)
        state.advance_to(decision_time)
        candidates = generator.generate("U1", decision_time).candidates
        builder = ContextBuilder(state)
        raw_contexts = tuple(
            builder.build("U1", candidate, decision_time) for candidate in candidates
        )
        scaler = FeatureScaler()
        scaled_contexts = scaler.fit_transform(raw_contexts)

        self.assertEqual(len(raw_contexts), 2)
        self.assertEqual(len(scaled_contexts), 2)
        self.assertTrue(all(len(context) == 15 for context in scaled_contexts))
        self.assertEqual(
            tuple(context[13:] for context in scaled_contexts),
            tuple(context[13:] for context in raw_contexts),
        )


if __name__ == "__main__":
    unittest.main()
