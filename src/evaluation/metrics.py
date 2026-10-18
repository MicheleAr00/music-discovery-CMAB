"""Track recommendation and artist-level relevance metrics for one phase.

K first selects the algorithm's Top-K tracks. Relevance evaluation then maps
those tracks to artists and removes duplicate artists, preserving the first
track's position. Precision, Recall, NDCG, and Artist Novelty use that ordered
artist list. Track Novelty and Catalog Coverage continue to use the original
track list; artist-level ILD applies the specified artist-identity distance to
all Top-K tracks. Reward remains the simulator's selected-action ``log1p``
feedback. Novelty flags come from pre-decision contexts, and phase coverage
uses the candidate catalog actually exposed at evaluated decision points.
Decision metrics are macro-averaged over users; coverage follows its
phase-level set-union definition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Iterable, Mapping, Optional

from src.data.state import PlayEvent
from src.simulation.reward import DEFAULT_REWARD_HORIZON, RewardCalculator
from src.simulation.simulator import DecisionRecord, SimulationResult


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """One metric value and the sample counts behind its aggregation."""

    value: Optional[float]
    users_used: int
    decisions_used: int
    aggregation: str


@dataclass(frozen=True, slots=True)
class OfflineMetricsResult:
    """Metrics for one phase, excluding logged decisions with empty pools."""

    phase: str
    metrics: Mapping[str, MetricSummary]
    k_values: tuple[int, ...]
    users_evaluated: int
    decisions_evaluated: int
    recall_excluded_decisions: int

    def __getitem__(self, metric_name: str) -> MetricSummary:
        """Look up a metric by its stable name, such as ``precision@10``."""
        return self.metrics[metric_name]


def _macro_user_summary(
    per_user: Mapping[str, list[float]], *, decisions_used: Optional[int] = None
) -> MetricSummary:
    user_means = [sum(values) / len(values) for values in per_user.values() if values]
    decision_count = (
        sum(len(values) for values in per_user.values())
        if decisions_used is None
        else decisions_used
    )
    return MetricSummary(
        value=sum(user_means) / len(user_means) if user_means else None,
        users_used=len(user_means),
        decisions_used=decision_count,
        aggregation="macro_user",
    )


def _artist_level_ranking(artist_keys: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Deduplicate a Top-K track ranking by artist, preserving first occurrence.

    The returned indices identify the first track representing each artist;
    those indices are also used to carry its pre-decision Artist Novelty flag.
    """
    seen: set[str] = set()
    ranked_artists: list[str] = []
    first_track_indices: list[int] = []
    for index, artist_key in enumerate(artist_keys):
        if artist_key in seen:
            continue
        seen.add(artist_key)
        ranked_artists.append(artist_key)
        first_track_indices.append(index)
    return tuple(ranked_artists), tuple(first_track_indices)


class OfflineMetricsEvaluator:
    """Compute ranking, novelty, diversity, coverage, and reward summaries.

    The future-event index is the same compact per-user/per-artist timestamp
    architecture used for simulator reward. It is queried only here, after
    the policy has produced its immutable decision log. The evaluator does not
    mutate that log or any policy state.

    Coverage denominators come from ``SimulationResult.eligible_catalog`` and
    ``eligible_artists``: the unions of deduplicated candidate pools at all
    usable decision points in this phase. Coverage is reported at every
    requested K, plus unsuffixed aliases for the largest requested K.
    """

    def __init__(
        self,
        events: Optional[Iterable[PlayEvent]] = None,
        *,
        horizon=DEFAULT_REWARD_HORIZON,
        reward_calculator: Optional[RewardCalculator] = None,
    ) -> None:
        if reward_calculator is not None:
            if events is not None:
                raise ValueError("provide events or reward_calculator, not both")
            self._future = reward_calculator
        elif events is not None:
            self._future = RewardCalculator(events, horizon=horizon)
        else:
            raise ValueError("events or reward_calculator is required")
        self._horizon = horizon

    def evaluate(
        self,
        simulation_result: SimulationResult,
        *,
        k_values: tuple[int, ...] = (5, 10, 20),
    ) -> OfflineMetricsResult:
        """Evaluate one phase using its stored Top-K tracks and raw flags.

        ``DecisionRecord`` must contain at least ``min(K, candidate_count)``
        ranked candidates for each requested K. Configure the simulator's
        ``top_k`` to at least the largest requested K when the candidate pool
        can be larger. Empty candidate lists have undefined precision and
        are excluded from all metric aggregation because no recommendation was
        possible. ``SimulationResult`` still retains those skipped decision
        records for auditing. NDCG is undefined when its ideal DCG is zero.

        K always refers to the first K tracks ranked by the policy. For
        artist-level metrics, only after taking that track prefix are artist
        keys deduplicated, preserving their first-occurrence order. Therefore
        the effective artist ranking can contain fewer than K entries.
        """
        if not isinstance(simulation_result, SimulationResult):
            raise TypeError("simulation_result must be a SimulationResult")
        if simulation_result.phase not in {"validation", "test"}:
            raise ValueError("metrics support validation or test results only")
        if not k_values:
            raise ValueError("k_values must contain at least one positive integer")
        normalized_k: list[int] = []
        for k in k_values:
            if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
                raise ValueError("every K must be a positive integer")
            if k not in normalized_k:
                normalized_k.append(k)
        ks = tuple(sorted(normalized_k))

        all_decisions = simulation_result.decisions
        if any(decision.phase != simulation_result.phase for decision in all_decisions):
            raise ValueError("all decisions must belong to the result phase")

        # Check that the stored ranking is long enough to answer every query.
        for decision in all_decisions:
            if decision.top_k_effective_size != len(decision.top_k_candidates):
                raise ValueError("top_k_effective_size does not match stored candidates")
            for k in ks:
                required = min(k, decision.candidate_count)
                if len(decision.top_k_candidates) < required:
                    raise ValueError(
                        f"decision at {decision.decision_time!s} stores only "
                        f"{len(decision.top_k_candidates)} ranked candidates; "
                        f"K={k} requires {required}. Increase simulator top_k."
                    )
            if decision.top_k_candidates and (
                len(decision.top_k_track_novelty) != len(decision.top_k_candidates)
                or len(decision.top_k_artist_novelty) != len(decision.top_k_candidates)
            ):
                raise ValueError(
                    "DecisionRecord is missing Top-K pre-decision novelty flags; "
                    "rerun the simulator with the metrics-compatible output"
                )

        decisions = tuple(
            decision for decision in all_decisions if decision.candidate_count > 0
        )

        per_metric: dict[str, dict[str, list[float]]] = {}

        def add(metric: str, user_id: str, value: float) -> None:
            per_metric.setdefault(metric, {}).setdefault(user_id, []).append(value)

        recall_excluded = 0
        eligible_recommendation_keys: dict[int, set[str]] = {
            k: set() for k in ks
        }
        eligible_recommendation_artists: dict[int, set[str]] = {
            k: set() for k in ks
        }

        for decision in decisions:
            relevant_artists = self._future.future_artist_keys(
                decision.user_id, decision.decision_time, self._horizon
            )
            artist_keys = tuple(
                candidate.track.artist_key for candidate in decision.top_k_candidates
            )

            if not relevant_artists:
                recall_excluded += 1
            for k in ks:
                prefix = min(k, len(decision.top_k_candidates))
                candidates = decision.top_k_candidates[:prefix]
                track_artist_keys = artist_keys[:prefix]
                ranked_artists, first_track_indices = _artist_level_ranking(
                    track_artist_keys
                )
                artist_relevance = tuple(
                    artist in relevant_artists for artist in ranked_artists
                )
                track_list_size = len(candidates)
                artist_list_size = len(ranked_artists)
                suffix = f"@{k}"

                # Track novelty counts every recommended track independently.
                if track_list_size:
                    add(
                        f"track_novelty{suffix}",
                        decision.user_id,
                        sum(decision.top_k_track_novelty[:prefix]) / track_list_size,
                    )

                # Precision and Artist Novelty use first-occurrence artists.
                if artist_list_size:
                    add(
                        f"precision{suffix}",
                        decision.user_id,
                        sum(artist_relevance) / artist_list_size,
                    )
                    add(
                        f"artist_novelty{suffix}",
                        decision.user_id,
                        sum(
                            decision.top_k_artist_novelty[index]
                            for index in first_track_indices
                        ) / artist_list_size,
                    )

                # Recall uses the distinct artists in the ordered ranking.
                if relevant_artists:
                    hits = sum(artist_relevance)
                    add(
                        f"recall{suffix}",
                        decision.user_id,
                        hits / len(relevant_artists),
                    )

                if relevant_artists and artist_list_size:
                    dcg = sum(
                        1.0 / math.log2(rank + 2)
                        for rank, relevant in enumerate(artist_relevance)
                        if relevant
                    )
                    # IDCG is based on all relevant artists available at the
                    # requested track cutoff, even when duplicate tracks make
                    # the deduplicated recommendation ranking shorter.
                    ideal_hits = min(k, len(relevant_artists))
                    idcg = sum(
                        1.0 / math.log2(rank + 2)
                        for rank in range(ideal_hits)
                    )
                    if idcg > 0.0:
                        # Binary artist relevance and a deduplicated artist
                        # ranking guarantee DCG <= IDCG, hence NDCG <= 1.
                        ndcg = min(1.0, dcg / idcg)
                        add(f"ndcg{suffix}", decision.user_id, ndcg)

                if track_list_size < 2:
                    ild = 0.0
                else:
                    counts: dict[str, int] = {}
                    for artist in track_artist_keys:
                        counts[artist] = counts.get(artist, 0) + 1
                    same_artist_pairs = sum(
                        count * (count - 1) for count in counts.values()
                    )
                    ild = 1.0 - same_artist_pairs / (
                        track_list_size * (track_list_size - 1)
                    )
                add(f"artist_ild{suffix}", decision.user_id, ild)

                eligible_recommendation_keys[k].update(
                    candidate.item_key for candidate in candidates
                )
                eligible_recommendation_artists[k].update(ranked_artists)

            if decision.reward is not None:
                add("average_reward", decision.user_id, decision.reward)

        summaries: dict[str, MetricSummary] = {
            name: _macro_user_summary(users) for name, users in per_metric.items()
        }

        evaluated_users = len({decision.user_id for decision in decisions})
        evaluated_decisions = len(decisions)
        for k in ks:
            summaries.setdefault(
                f"precision@{k}", _macro_user_summary({})
            )
            summaries.setdefault(f"recall@{k}", _macro_user_summary({}))
            summaries.setdefault(f"ndcg@{k}", _macro_user_summary({}))
            summaries.setdefault(f"track_novelty@{k}", _macro_user_summary({}))
            summaries.setdefault(f"artist_novelty@{k}", _macro_user_summary({}))
            summaries.setdefault(f"artist_ild@{k}", _macro_user_summary({}))
            catalog_denominator = len(simulation_result.eligible_catalog)
            artist_denominator = len(simulation_result.eligible_artists)
            summaries[f"catalog_coverage@{k}"] = MetricSummary(
                value=(
                    len(eligible_recommendation_keys[k]) / catalog_denominator
                    if catalog_denominator
                    else None
                ),
                users_used=evaluated_users,
                decisions_used=evaluated_decisions,
                aggregation="phase_union",
            )
            summaries[f"artist_coverage@{k}"] = MetricSummary(
                value=(
                    len(eligible_recommendation_artists[k]) / artist_denominator
                    if artist_denominator
                    else None
                ),
                users_used=evaluated_users,
                decisions_used=evaluated_decisions,
                aggregation="phase_union",
            )

        # Coverage is intrinsically a set-union metric. Unsuffixed names are
        # aliases for the largest requested K, while every K remains explicit.
        largest_k = ks[-1]
        summaries["catalog_coverage"] = summaries[f"catalog_coverage@{largest_k}"]
        summaries["artist_coverage"] = summaries[f"artist_coverage@{largest_k}"]
        summaries.setdefault("average_reward", _macro_user_summary({}))

        return OfflineMetricsResult(
            phase=simulation_result.phase,
            metrics=MappingProxyType(summaries),
            k_values=ks,
            users_evaluated=evaluated_users,
            decisions_evaluated=evaluated_decisions,
            recall_excluded_decisions=recall_excluded,
        )
