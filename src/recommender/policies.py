"""Interchangeable ranking policies for the sequential recommendation engine.

Every policy ranks the exact candidate sequence supplied by the simulator.
Policies return one selected Top-1 action and a Top-K evaluation ranking;
only the selected action is associated with delayed reward feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import random
from numbers import Real
from typing import Optional, Protocol, Sequence

from src.data.state import TemporalState
from src.simulation.candidate_generator import Candidate
from src.simulation.context import ContextVector
from src.simulation.linucb import LinUCB


DEFAULT_EPSILONS: tuple[float, ...] = (0.05, 0.10, 0.20)


@dataclass(frozen=True, slots=True)
class PolicyRanking:
    """Top-1 action and its ordered Top-K view for metric evaluation."""

    selected_index: int
    top_k_indices: tuple[int, ...]
    top_k_scores: tuple[float, ...]
    selected_mean: Optional[float] = None
    selected_exploration: Optional[float] = None


class CandidatePolicy(Protocol):
    """Policy contract used by the shared simulator."""

    def rank(
        self,
        candidates: Sequence[Candidate],
        *,
        user_id: str,
        timestamp: datetime,
        state: TemporalState,
        contexts: Sequence[ContextVector],
        top_k: int,
    ) -> PolicyRanking:
        """Select Top-1 and return a Top-K ordering over supplied candidates."""

    def update(self, context: ContextVector, reward: float) -> None:
        """Receive reward only when its delayed feedback becomes available."""

    def reset(self) -> None:
        """Reset learned state at a validation/test boundary if applicable."""


def _validate_ranking_input(
    candidates: Sequence[Candidate], top_k: int, state: TemporalState, timestamp: datetime
) -> None:
    if not candidates:
        raise ValueError("cannot rank an empty candidate pool")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if state.current_time != timestamp:
        raise ValueError("TemporalState must be at the exact decision timestamp")


def _make_ranking(
    order: Sequence[int], scores: Sequence[float], top_k: int
) -> PolicyRanking:
    indices = tuple(order[:top_k])
    if not indices:
        raise ValueError("policy ranking must contain a selected candidate")
    return PolicyRanking(
        selected_index=indices[0],
        top_k_indices=indices,
        top_k_scores=tuple(float(scores[index]) for index in indices),
    )


class RandomPolicy:
    """Uniform random selection and a seeded random permutation for Top-K."""

    def __init__(self, random_seed: int = 0) -> None:
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise ValueError("random_seed must be an integer")
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)

    def rank(
        self,
        candidates: Sequence[Candidate],
        *,
        user_id: str,
        timestamp: datetime,
        state: TemporalState,
        contexts: Sequence[ContextVector],
        top_k: int,
    ) -> PolicyRanking:
        _validate_ranking_input(candidates, top_k, state, timestamp)
        order = self._rng.sample(range(len(candidates)), len(candidates))
        # Random policy deliberately ignores state, contexts, and rewards.
        return _make_ranking(order, (0.0,) * len(candidates), top_k)

    def update(self, context: ContextVector, reward: float) -> None:
        """Random has no learned parameters and ignores delayed reward."""

    def reset(self) -> None:
        """Keep the isolated RNG stream continuous across phase boundaries."""


class GlobalPopularityPolicy:
    """Rank candidate tracks by their artist's global plays before time t.

    Popularity uses ``TemporalState.get_artist_play_count`` in accordance
    with P_a(t) in the experiment specification. Tracks tied on artist
    popularity are ordered by canonical track key.
    """

    def rank(
        self,
        candidates: Sequence[Candidate],
        *,
        user_id: str,
        timestamp: datetime,
        state: TemporalState,
        contexts: Sequence[ContextVector],
        top_k: int,
    ) -> PolicyRanking:
        _validate_ranking_input(candidates, top_k, state, timestamp)
        scores = tuple(
            float(state.get_artist_play_count(candidate.track.artist_key))
            for candidate in candidates
        )
        order = sorted(
            range(len(candidates)),
            key=lambda index: (-scores[index], candidates[index].item_key),
        )
        return _make_ranking(order, scores, top_k)

    def update(self, context: ContextVector, reward: float) -> None:
        """Popularity is an observation-only baseline; reward is ignored."""

    def reset(self) -> None:
        """There are no learned parameters to reset."""


class PersonalizedGreedyPolicy:
    """Exploit user-artist history, then track popularity, then track key."""

    @staticmethod
    def _scores(
        candidates: Sequence[Candidate], user_id: str, state: TemporalState
    ) -> tuple[float, ...]:
        return tuple(
            float(state.get_user_artist_count(user_id, candidate.track.artist_key))
            for candidate in candidates
        )

    @staticmethod
    def _order(
        candidates: Sequence[Candidate],
        user_id: str,
        state: TemporalState,
        scores: Sequence[float],
    ) -> list[int]:
        return sorted(
            range(len(candidates)),
            key=lambda index: (
                -scores[index],
                -state.get_item_play_count(candidates[index].item_key),
                candidates[index].item_key,
            ),
        )

    def rank(
        self,
        candidates: Sequence[Candidate],
        *,
        user_id: str,
        timestamp: datetime,
        state: TemporalState,
        contexts: Sequence[ContextVector],
        top_k: int,
    ) -> PolicyRanking:
        _validate_ranking_input(candidates, top_k, state, timestamp)
        scores = self._scores(candidates, user_id, state)
        order = self._order(candidates, user_id, state, scores)
        return _make_ranking(order, scores, top_k)

    def update(self, context: ContextVector, reward: float) -> None:
        """Greedy history scoring does not learn from reward feedback."""

    def reset(self) -> None:
        """There are no learned parameters to reset."""


class EpsilonGreedyPolicy(PersonalizedGreedyPolicy):
    """Seeded epsilon exploration over the Personalized Greedy baseline.

    One branch is drawn per decision. On exploration, the Top-K output is a
    random permutation with its first item selected as Top-1. On exploitation,
    the complete ranking is Personalized Greedy's deterministic ordering.
    """

    def __init__(self, epsilon: float, random_seed: int = 0) -> None:
        if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
            raise ValueError("epsilon must be a finite number in [0, 1]")
        epsilon_value = float(epsilon)
        if not math.isfinite(epsilon_value) or not 0.0 <= epsilon_value <= 1.0:
            raise ValueError("epsilon must be a finite number in [0, 1]")
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise ValueError("random_seed must be an integer")
        self.epsilon = epsilon_value
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)

    def rank(
        self,
        candidates: Sequence[Candidate],
        *,
        user_id: str,
        timestamp: datetime,
        state: TemporalState,
        contexts: Sequence[ContextVector],
        top_k: int,
    ) -> PolicyRanking:
        _validate_ranking_input(candidates, top_k, state, timestamp)
        if self._rng.random() < self.epsilon:
            order = self._rng.sample(range(len(candidates)), len(candidates))
            return _make_ranking(order, (0.0,) * len(candidates), top_k)

        scores = self._scores(candidates, user_id, state)
        order = self._order(candidates, user_id, state, scores)
        return _make_ranking(order, scores, top_k)


class LinUCBPolicy:
    """Adapter exposing the existing LinUCB learner through CandidatePolicy."""

    def __init__(self, learner: LinUCB) -> None:
        self._learner = learner

    @property
    def learner(self) -> LinUCB:
        """The current phase's underlying LinUCB model."""
        return self._learner

    def rank(
        self,
        candidates: Sequence[Candidate],
        *,
        user_id: str,
        timestamp: datetime,
        state: TemporalState,
        contexts: Sequence[ContextVector],
        top_k: int,
    ) -> PolicyRanking:
        _validate_ranking_input(candidates, top_k, state, timestamp)
        if len(contexts) != len(candidates):
            raise ValueError("one context is required for each candidate")
        scored = self._learner.score_batch(
            contexts,
            candidate_ids=(candidate.item_key for candidate in candidates),
        )
        order = sorted(scored, key=lambda row: (-row.score, row.index))[:top_k]
        selected = order[0]
        return PolicyRanking(
            selected_index=selected.index,
            top_k_indices=tuple(row.index for row in order),
            top_k_scores=tuple(row.score for row in order),
            selected_mean=selected.mean,
            selected_exploration=selected.exploration,
        )

    def update(self, context: ContextVector, reward: float) -> None:
        self._learner.update(context, reward)

    def reset(self) -> None:
        """Create a fresh A=I, b=0 learner while keeping its configuration."""
        self._learner = LinUCB(
            n_features=self._learner.n_features,
            alpha=self._learner.alpha,
        )
