"""Tests for the indexed future observed-interaction reward."""

from datetime import datetime, timedelta, timezone
import math
import unittest

from src.data.state import PlayEvent
from src.simulation.reward import DEFAULT_REWARD_HORIZON, RewardCalculator


BASE = datetime(2005, 10, 30, 10, 0)


def event(
    user: str,
    seconds: int,
    artist: str = "A1",
    track: str = "T1",
) -> PlayEvent:
    return PlayEvent(user, BASE + timedelta(seconds=seconds), track, artist)


def naive_count(events, user, artist, decision_time, horizon):
    """Reference definition: ``user/artist`` matches and ``t < time <= end``."""
    end = decision_time + horizon
    return sum(
        1
        for item in events
        if item.user_id == user
        and item.artist_id == artist
        and decision_time < item.timestamp <= end
    )


class RewardCalculatorTest(unittest.TestCase):
    def test_zero_future_interactions_return_zero_count_and_reward(self) -> None:
        calculator = RewardCalculator([event("U1", 0, "A1")])
        self.assertEqual(calculator.future_count("U1", "A2", BASE), 0)
        self.assertEqual(calculator.reward("U1", "A2", BASE), 0.0)

    def test_one_future_interaction_returns_log_two(self) -> None:
        calculator = RewardCalculator([event("U1", 1)])
        self.assertEqual(calculator.future_count("U1", "A1", BASE), 1)
        self.assertAlmostEqual(calculator.reward("U1", "A1", BASE), math.log(2))

    def test_multiple_future_interactions_return_exact_count(self) -> None:
        events = [event("U1", second) for second in (1, 2, 3, 4)]
        calculator = RewardCalculator(events)
        self.assertEqual(calculator.future_count("U1", "A1", BASE), 4)
        self.assertAlmostEqual(calculator.reward("U1", "A1", BASE), math.log(5))

    def test_lower_boundary_is_excluded_and_upper_boundary_included(self) -> None:
        horizon = DEFAULT_REWARD_HORIZON
        events = [
            PlayEvent("U1", BASE, "T_AT_T", "A1"),
            PlayEvent("U1", BASE + timedelta(days=7), "T_AT_END", "A1"),
        ]
        calculator = RewardCalculator(events)
        self.assertEqual(calculator.future_count("U1", "A1", BASE, horizon), 1)
        self.assertAlmostEqual(calculator.reward("U1", "A1", BASE, horizon), math.log(2))

    def test_event_just_outside_horizon_is_excluded(self) -> None:
        events = [
            PlayEvent(
                "U1",
                BASE + timedelta(days=7, microseconds=1),
                "T1",
                "A1",
            )
        ]
        calculator = RewardCalculator(events)
        self.assertEqual(calculator.future_count("U1", "A1", BASE), 0)

    def test_past_events_are_not_counted(self) -> None:
        calculator = RewardCalculator([event("U1", -1), event("U1", 0)])
        self.assertEqual(calculator.future_count("U1", "A1", BASE), 0)

    def test_user_isolation(self) -> None:
        calculator = RewardCalculator([event("U2", 1, "A1"), event("U1", 2, "A1")])
        self.assertEqual(calculator.future_count("U1", "A1", BASE), 1)

    def test_artist_isolation(self) -> None:
        calculator = RewardCalculator([event("U1", 1, "A2"), event("U1", 2, "A1")])
        self.assertEqual(calculator.future_count("U1", "A1", BASE), 1)

    def test_artist_reward_aggregates_different_tracks(self) -> None:
        events = [event("U1", 1, "A1", "T1"), event("U1", 2, "A1", "T2")]
        calculator = RewardCalculator(events)
        self.assertEqual(calculator.future_count("U1", "A1", BASE), 2)
        self.assertAlmostEqual(calculator.reward("U1", "A1", BASE), math.log(3))

    def test_multiple_artists_do_not_contaminate_requested_reward(self) -> None:
        events = [
            event("U1", 1, "A1", "T1"),
            event("U1", 2, "A2", "T2"),
            event("U1", 3, "A1", "T3"),
            event("U1", 4, "A3", "T4"),
        ]
        calculator = RewardCalculator(events)
        self.assertEqual(calculator.future_count("U1", "A1", BASE), 2)
        self.assertEqual(calculator.future_count("U1", "A2", BASE), 1)

    def test_logarithmic_reward_for_several_counts(self) -> None:
        for expected_count in (0, 1, 2, 5, 10):
            with self.subTest(count=expected_count):
                events = [event("U1", second) for second in range(1, expected_count + 1)]
                calculator = RewardCalculator(events)
                self.assertEqual(
                    calculator.future_count("U1", "A1", BASE), expected_count
                )
                self.assertAlmostEqual(
                    calculator.reward("U1", "A1", BASE),
                    math.log(1 + expected_count),
                )

    def test_repeated_queries_are_deterministic_and_do_not_mutate_index(self) -> None:
        calculator = RewardCalculator([event("U1", 1), event("U1", 2)])
        before = {
            key: tuple(timestamps) for key, timestamps in calculator._timestamps.items()
        }
        rewards = [calculator.reward("U1", "A1", BASE) for _ in range(4)]
        after = {
            key: tuple(timestamps) for key, timestamps in calculator._timestamps.items()
        }
        self.assertEqual(rewards, [math.log(3)] * 4)
        self.assertEqual(before, after)

    def test_index_matches_naive_reference_over_users_artists_and_horizons(self) -> None:
        events = [
            event("U1", -3, "A1", "T0"),
            event("U1", 0, "A1", "T1"),
            event("U1", 1, "A1", "T1"),
            event("U2", 2, "A1", "T1"),
            event("U1", 3, "A2", "T2"),
            event("U1", 5, "A1", "T3"),
            event("U1", 10, "A2", "T2"),
            event("U2", 12, "A2", "T2"),
            event("U1", 20, "A1", "T4"),
        ]
        calculator = RewardCalculator(events)
        for user in ("U1", "U2", "missing"):
            for artist in ("A1", "A2", "missing"):
                for at_seconds, horizon_seconds in ((0, 4), (2, 10), (-5, 30), (20, 1)):
                    decision_time = BASE + timedelta(seconds=at_seconds)
                    horizon = timedelta(seconds=horizon_seconds)
                    with self.subTest(user=user, artist=artist, at=at_seconds):
                        self.assertEqual(
                            calculator.future_count(
                                user, artist, decision_time, horizon
                            ),
                            naive_count(events, user, artist, decision_time, horizon),
                        )

    def test_non_default_horizon_and_query_override(self) -> None:
        events = [event("U1", 1), event("U1", 5)]
        calculator = RewardCalculator(events, horizon=timedelta(seconds=2))
        self.assertEqual(calculator.future_count("U1", "A1", BASE), 1)
        self.assertEqual(
            calculator.future_count("U1", "A1", BASE, timedelta(seconds=10)), 2
        )

    def test_invalid_horizons_are_rejected(self) -> None:
        for horizon in (timedelta(0), timedelta(seconds=-1), 0, None):
            with self.subTest(horizon=horizon):
                with self.assertRaisesRegex(ValueError, "positive timedelta"):
                    RewardCalculator([], horizon=horizon)
        calculator = RewardCalculator([])
        with self.assertRaisesRegex(ValueError, "positive timedelta"):
            calculator.future_count("U1", "A1", BASE, timedelta(0))

    def test_identity_fallback_uses_shared_artist_key_convention(self) -> None:
        fallback_artist_key = "name::Fallback Artist"
        events = [event("U1", 1, fallback_artist_key)]
        calculator = RewardCalculator(events)
        self.assertEqual(
            calculator.future_count(
                "U1", None, BASE, artist_name="Fallback Artist"
            ),
            1,
        )
        self.assertEqual(
            calculator.reward("U1", None, BASE, artist_name="Fallback Artist"),
            math.log(2),
        )

    def test_invalid_query_id_or_timestamp_is_rejected(self) -> None:
        calculator = RewardCalculator([event("U1", 1)])
        for user_id, artist_id, decision_time in (
            (" ", "A1", BASE),
            ("U1", None, BASE),
            ("U1", " ", BASE),
            ("U1", "A1", "2005-10-30"),
        ):
            with self.subTest(user=user_id, artist=artist_id, time=decision_time):
                with self.assertRaises((ValueError, TypeError)):
                    calculator.future_count(user_id, artist_id, decision_time)

    def test_invalid_and_inconsistent_events_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "PlayEvent"):
            RewardCalculator([object()])
        with self.assertRaisesRegex(ValueError, "ordered"):
            RewardCalculator([event("U1", 2), event("U1", 1)])
        aware_event = PlayEvent(
            "U1", BASE.replace(tzinfo=timezone.utc), "T1", "A1"
        )
        with self.assertRaisesRegex(ValueError, "consistent timezone"):
            RewardCalculator([event("U1", 0), aware_event])

    def test_decision_time_must_match_index_timezone_convention(self) -> None:
        event_time = BASE.replace(tzinfo=timezone.utc)
        calculator = RewardCalculator([PlayEvent("U1", event_time, "T1", "A1")])
        with self.assertRaisesRegex(ValueError, "same timezone convention"):
            calculator.future_count("U1", "A1", BASE)


if __name__ == "__main__":
    unittest.main()
