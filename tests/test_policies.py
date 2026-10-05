"""Baseline policy and interchangeable-simulator tests."""

from datetime import datetime, timedelta
from unittest.mock import patch
import unittest

from src.data.state import PlayEvent, TemporalState
from src.recommender.policies import (
    DEFAULT_EPSILONS,
    EpsilonGreedyPolicy,
    GlobalPopularityPolicy,
    LinUCBPolicy,
    PersonalizedGreedyPolicy,
    RandomPolicy,
)
from src.simulation.candidate_generator import Candidate, CandidateGenerator, TrackMetadata
from src.simulation.context import ContextBuilder
from src.simulation.linucb import LinUCB
from src.simulation.reward import RewardCalculator
from src.simulation.scaler import FeatureScaler
from src.simulation.simulator import SequentialOfflineBanditSimulator


T = datetime(2009, 1, 1, 10, 0)
BASE = datetime(2005, 10, 30, 10, 0)
HORIZON = timedelta(days=7)
CONTEXT = (0.0,) * 15


def event(user: str, when: datetime, track: str, artist: str) -> PlayEvent:
    return PlayEvent(user, when, track, artist)


def candidate(track: str, artist: str) -> Candidate:
    return Candidate(
        TrackMetadata(track, track, artist, artist),
        frozenset(),
    )


def state_at(events: list[PlayEvent], timestamp: datetime = T):
    state = TemporalState(events)
    generator = CandidateGenerator(state)
    state.advance_to(timestamp)
    return state, generator


def candidate_keys(rows):
    return tuple(row.track_key for row in rows)


def basic_policy_events() -> list[PlayEvent]:
    return [
        event("U", T - timedelta(minutes=4), "TA", "A"),
        event("U", T - timedelta(minutes=3), "TB", "B"),
    ]


def build_simulator(policy, *, top_k=3):
    events = [
        event("U", BASE, "W0", "WA"),
        event("U", BASE + timedelta(minutes=20), "W1", "WB"),
        event("U", T, "D0", "DA"),
        event("U", T + HORIZON, "F0", "DA"),
    ]
    state = TemporalState(events)
    return SequentialOfflineBanditSimulator(
        events=events,
        eligible_users={"U"},
        state=state,
        candidate_generator=CandidateGenerator(state),
        context_builder=ContextBuilder(state),
        scaler=FeatureScaler(),
        reward_calculator=RewardCalculator(events, horizon=HORIZON),
        policy=policy,
        candidate_pool_depth=50,
        top_k=top_k,
        reward_horizon=HORIZON,
    )


def prepare_simulator(simulator):
    simulator.prepare_warmup(BASE - timedelta(seconds=1), BASE + timedelta(days=1))


class RandomPolicyTests(unittest.TestCase):
    def test_selects_only_pool_candidates_without_top_k_duplicates(self):
        policy = RandomPolicy(random_seed=42)
        pool = (candidate("T1", "A1"), candidate("T2", "A2"), candidate("T3", "A3"))
        state, _ = state_at([])
        ranked = policy.rank(
            pool, user_id="U", timestamp=T, state=state, contexts=(), top_k=3
        )
        self.assertEqual(len(ranked.top_k_indices), 3)
        self.assertEqual(len(set(ranked.top_k_indices)), 3)
        self.assertTrue(set(ranked.top_k_indices).issubset(range(len(pool))))
        self.assertEqual(ranked.selected_index, ranked.top_k_indices[0])

    def test_random_ignores_user_context_and_reward_related_state(self):
        pool = (candidate("T1", "A1"), candidate("T2", "A2"))
        ordinary_state, _ = state_at([])
        history_state, _ = state_at(
            [event("U", T - timedelta(minutes=1), "T1", "A1")]
        )
        first = RandomPolicy(17).rank(
            pool, user_id="U1", timestamp=T, state=ordinary_state,
            contexts=(CONTEXT,), top_k=2,
        )
        second = RandomPolicy(17).rank(
            pool, user_id="U2", timestamp=T, state=history_state,
            contexts=((99.0,) * 15,), top_k=2,
        )
        self.assertEqual(first, second)

    def test_reproducible_with_same_seed(self):
        pool = tuple(candidate(f"T{i}", f"A{i}") for i in range(20))
        state, _ = state_at([])
        outputs = []
        for _ in range(2):
            policy = RandomPolicy(random_seed=42)
            outputs.append(
                tuple(
                    policy.rank(
                        pool, user_id="U", timestamp=T, state=state,
                        contexts=(), top_k=5,
                    )
                    for _ in range(5)
                )
            )
        self.assertEqual(outputs[0], outputs[1])

    def test_top_k_is_limited_to_smaller_candidate_pool(self):
        pool = (candidate("T1", "A1"), candidate("T2", "A2"))
        state, _ = state_at([])
        ranked = RandomPolicy(9).rank(
            pool, user_id="U", timestamp=T, state=state, contexts=(), top_k=10
        )
        self.assertEqual(len(ranked.top_k_indices), 2)


class PopularityPolicyTests(unittest.TestCase):
    def test_ranks_by_predecision_global_artist_play_count(self):
        events = [
            event("U1", T - timedelta(minutes=6), "TA", "A"),
            event("U2", T - timedelta(minutes=5), "TA2", "A"),
            event("U3", T - timedelta(minutes=4), "TA3", "A"),
            event("U1", T - timedelta(minutes=3), "TB", "B"),
            event("U2", T - timedelta(minutes=2), "TB", "B"),
        ]
        state, _ = state_at(events)
        pool = (candidate("TB", "B"), candidate("TA3", "A"), candidate("TA", "A"))
        ranked = GlobalPopularityPolicy().rank(
            pool, user_id="U1", timestamp=T, state=state, contexts=(), top_k=3
        )
        self.assertEqual(
            candidate_keys(tuple(pool[i] for i in ranked.top_k_indices)),
            ("TA", "TA3", "TB"),
        )
        self.assertEqual(ranked.top_k_scores, (3.0, 3.0, 2.0))

    def test_future_popularity_event_cannot_change_ranking_at_t(self):
        prefix = basic_policy_events()
        with_future = prefix + [event("U", T + timedelta(minutes=30), "TB", "B")]
        pool = (candidate("TA", "A"), candidate("TB", "B"))
        rankings = []
        for events in (prefix, with_future):
            state, _ = state_at(events)
            ranking = GlobalPopularityPolicy().rank(
                pool, user_id="U", timestamp=T, state=state, contexts=(), top_k=2
            )
            rankings.append(ranking)
        self.assertEqual(rankings[0], rankings[1])

    def test_ranking_updates_when_a_new_play_becomes_observable(self):
        events = [
            event("U", T - timedelta(minutes=3), "TA", "A"),
            event("U", T - timedelta(minutes=2), "TB", "B"),
            event("U", T, "TB", "B"),
        ]
        state, _ = state_at(events)
        pool = (candidate("TA", "A"), candidate("TB", "B"))
        policy = GlobalPopularityPolicy()
        before = policy.rank(pool, user_id="U", timestamp=T, state=state, contexts=(), top_k=2)
        state.advance_to(T + timedelta(microseconds=1))
        after = policy.rank(
            pool, user_id="U", timestamp=T + timedelta(microseconds=1),
            state=state, contexts=(), top_k=2,
        )
        self.assertEqual(pool[before.selected_index].track_key, "TA")
        self.assertEqual(pool[after.selected_index].track_key, "TB")


class PersonalizedGreedyPolicyTests(unittest.TestCase):
    def test_ranks_by_user_artist_count_and_unheard_artist_scores_zero(self):
        events = [
            event("U", T - timedelta(minutes=5), "TA1", "A"),
            event("U", T - timedelta(minutes=4), "TA2", "A"),
            event("U", T - timedelta(minutes=3), "TB", "B"),
        ]
        state, _ = state_at(events)
        pool = (candidate("TC", "C"), candidate("TB", "B"), candidate("TA", "A"))
        ranked = PersonalizedGreedyPolicy().rank(
            pool, user_id="U", timestamp=T, state=state, contexts=(), top_k=3
        )
        self.assertEqual(
            candidate_keys(tuple(pool[index] for index in ranked.top_k_indices)),
            ("TA", "TB", "TC"),
        )
        self.assertEqual(ranked.top_k_scores, (2.0, 1.0, 0.0))

    def test_deterministic_tie_break_is_track_popularity_then_key(self):
        events = [
            event("U", T - timedelta(minutes=6), "TA", "A"),
            event("U", T - timedelta(minutes=5), "TB", "B"),
            event("U", T - timedelta(minutes=4), "TB", "B"),
            event("U", T - timedelta(minutes=3), "TB", "B"),
        ]
        state, _ = state_at(events)
        pool = (candidate("Z_A", "A"), candidate("B_TRACK", "B"), candidate("A_TRACK", "A"))
        policy = PersonalizedGreedyPolicy()
        first = policy.rank(pool, user_id="U", timestamp=T, state=state, contexts=(), top_k=3)
        second = policy.rank(pool, user_id="U", timestamp=T, state=state, contexts=(), top_k=3)
        self.assertEqual(first, second)
        self.assertEqual(pool[first.selected_index].track_key, "B_TRACK")
        self.assertEqual(pool[first.top_k_indices[1]].track_key, "A_TRACK")

    def test_future_user_play_does_not_change_greedy_ranking_at_t(self):
        prefix = [event("U", T - timedelta(minutes=1), "TA", "A")]
        with_future = prefix + [event("U", T + timedelta(minutes=30), "TB", "B")]
        pool = (candidate("TA", "A"), candidate("TB", "B"))
        rankings = []
        for events in (prefix, with_future):
            state, _ = state_at(events)
            rankings.append(
                PersonalizedGreedyPolicy().rank(
                    pool, user_id="U", timestamp=T, state=state, contexts=(), top_k=2
                )
            )
        self.assertEqual(rankings[0], rankings[1])


class EpsilonGreedyPolicyTests(unittest.TestCase):
    def test_epsilon_zero_matches_personalized_greedy(self):
        state, _ = state_at(basic_policy_events())
        pool = (candidate("TB", "B"), candidate("TA", "A"))
        greedy = PersonalizedGreedyPolicy().rank(
            pool, user_id="U", timestamp=T, state=state, contexts=(), top_k=2
        )
        epsilon_zero = EpsilonGreedyPolicy(0.0, random_seed=42).rank(
            pool, user_id="U", timestamp=T, state=state, contexts=(), top_k=2
        )
        self.assertEqual(epsilon_zero, greedy)

    def test_epsilon_one_always_returns_random_pool_permutations(self):
        pool = tuple(candidate(f"T{i}", f"A{i}") for i in range(6))
        state, _ = state_at([])
        policy = EpsilonGreedyPolicy(1.0, random_seed=42)
        outputs = [
            policy.rank(pool, user_id="U", timestamp=T, state=state, contexts=(), top_k=4)
            for _ in range(12)
        ]
        self.assertTrue(all(row.selected_index == row.top_k_indices[0] for row in outputs))
        self.assertTrue(all(len(set(row.top_k_indices)) == 4 for row in outputs))
        self.assertGreater(len({row.selected_index for row in outputs}), 1)

    def test_intermediate_epsilon_and_seed_are_reproducible(self):
        self.assertEqual(DEFAULT_EPSILONS, (0.05, 0.10, 0.20))
        pool = tuple(candidate(f"T{i}", f"A{i}") for i in range(8))
        state, _ = state_at(basic_policy_events())
        runs = []
        for _ in range(2):
            policy = EpsilonGreedyPolicy(0.10, random_seed=2026)
            runs.append(
                tuple(
                    policy.rank(
                        pool, user_id="U", timestamp=T, state=state,
                        contexts=(), top_k=4,
                    )
                    for _ in range(30)
                )
            )
        self.assertEqual(runs[0], runs[1])

    def test_epsilon_greedy_does_not_see_future_user_history(self):
        prefix = [event("U", T - timedelta(minutes=1), "TA", "A")]
        with_future = prefix + [event("U", T + timedelta(minutes=30), "TB", "B")]
        pool = (candidate("TA", "A"), candidate("TB", "B"))
        outputs = []
        for events in (prefix, with_future):
            state, _ = state_at(events)
            policy = EpsilonGreedyPolicy(0.0, random_seed=3)
            outputs.append(
                policy.rank(pool, user_id="U", timestamp=T, state=state, contexts=(), top_k=2)
            )
        self.assertEqual(outputs[0], outputs[1])

    def test_delayed_reward_does_not_update_any_baseline_policy(self):
        policies = (
            RandomPolicy(1),
            GlobalPopularityPolicy(),
            PersonalizedGreedyPolicy(),
            EpsilonGreedyPolicy(0.10, random_seed=1),
        )
        rng_states = tuple(
            getattr(policy, "_rng").getstate() if hasattr(policy, "_rng") else None
            for policy in policies
        )
        for policy in policies:
            policy.update(CONTEXT, 7.25)
        after = tuple(
            getattr(policy, "_rng").getstate() if hasattr(policy, "_rng") else None
            for policy in policies
        )
        self.assertEqual(rng_states, after)


class SharedSimulatorPolicyTests(unittest.TestCase):
    def test_different_policies_receive_the_same_pool_and_generator_runs_once(self):
        observed_pools = []
        for policy in (
            RandomPolicy(5),
            GlobalPopularityPolicy(),
            LinUCBPolicy(LinUCB(alpha=0.2)),
        ):
            simulator = build_simulator(policy)
            prepare_simulator(simulator)
            generate = simulator._candidate_generator.generate
            policy_rank = policy.rank
            seen = []

            def capture_generate(user_id, timestamp, pool_depth=50):
                pools = generate(user_id, timestamp, pool_depth)
                seen.append((user_id, timestamp, candidate_keys(pools.candidates)))
                return pools

            def capture_rank(candidates, **kwargs):
                seen.append((kwargs["user_id"], kwargs["timestamp"], candidate_keys(candidates)))
                return policy_rank(candidates, **kwargs)

            with (
                patch.object(simulator._candidate_generator, "generate", side_effect=capture_generate) as gen_spy,
                patch.object(policy, "rank", side_effect=capture_rank) as policy_spy,
            ):
                phase = simulator.run_phase("validation", T, T + HORIZON)

            self.assertEqual(gen_spy.call_count, len(phase.decisions))
            self.assertEqual(policy_spy.call_count, len(phase.decisions))
            self.assertEqual(
                [row for row in seen[::2]], [row for row in seen[1::2]]
            )
            observed_pools.append([row[2] for row in seen[1::2]])
        self.assertEqual(observed_pools[0], observed_pools[1])

    def test_top1_alone_gets_one_delayed_reward_and_baselines_do_not_learn(self):
        policy = RandomPolicy(random_seed=7)
        simulator = build_simulator(policy, top_k=3)
        prepare_simulator(simulator)
        phase = simulator.run_phase("validation", T, T + HORIZON)
        self.assertEqual(len(phase.decisions), 1)
        self.assertEqual(phase.decisions[0].top_k_effective_size, 2)
        self.assertEqual(phase.feedback_scheduled, 1)
        self.assertEqual(phase.feedback_applied, 1)
        self.assertEqual(phase.feedback_pending_at_end, 0)
        self.assertEqual(phase.decisions[0].reward_available_time, T + HORIZON)


if __name__ == "__main__":
    unittest.main()
