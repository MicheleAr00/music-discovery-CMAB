"""Numerically stable, action-level LinUCB policy learner.

Scoring and selection are read-only. Only an explicit ``update(context,
reward)`` changes the model, and callers should invoke it for the selected
Top-1 action once its reward is available. Top-K is a ranked evaluation output,
not a set of observed actions and never triggers model updates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Hashable, Iterable, Optional


DEFAULT_N_FEATURES = 15


@dataclass(frozen=True, slots=True)
class LinUCBScore:
    """Score decomposition for one action in a scored candidate batch."""

    index: int
    candidate_id: Optional[Hashable]
    mean: float
    exploration: float
    score: float


def _validated_vector(context: Iterable[Real], n_features: int) -> tuple[float, ...]:
    """Convert a context to finite floats and validate its dimension."""
    try:
        values = tuple(context)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("context must be an iterable of numeric values") from exc
    if len(values) != n_features:
        raise ValueError(
            f"context must contain exactly {n_features} features; got {len(values)}"
        )
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise ValueError("context must contain only numeric values")
    vector = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("context values must all be finite (no NaN or infinity)")
    return vector


def _cholesky(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    """Return the lower Cholesky factor of a symmetric positive-definite matrix."""
    dimension = len(matrix)
    lower = [[0.0] * dimension for _ in range(dimension)]
    for row in range(dimension):
        for column in range(row + 1):
            residual = matrix[row][column] - math.fsum(
                lower[row][index] * lower[column][index]
                for index in range(column)
            )
            if row == column:
                if not math.isfinite(residual) or residual <= 0.0:
                    raise ArithmeticError("LinUCB matrix is not positive definite")
                lower[row][column] = math.sqrt(residual)
            else:
                lower[row][column] = residual / lower[column][column]
    return tuple(tuple(row) for row in lower)


def _solve_lower(
    lower: tuple[tuple[float, ...], ...], right_hand_side: tuple[float, ...]
) -> tuple[float, ...]:
    """Solve ``L y = rhs`` for lower-triangular ``L``."""
    solution: list[float] = []
    for row in range(len(lower)):
        residual = right_hand_side[row] - math.fsum(
            lower[row][column] * solution[column] for column in range(row)
        )
        solution.append(residual / lower[row][row])
    return tuple(solution)


def _solve_upper_from_lower(
    lower: tuple[tuple[float, ...], ...], right_hand_side: tuple[float, ...]
) -> tuple[float, ...]:
    """Solve ``L.T x = rhs`` for lower-triangular ``L``."""
    solution = [0.0] * len(lower)
    for row in range(len(lower) - 1, -1, -1):
        residual = right_hand_side[row] - math.fsum(
            lower[column][row] * solution[column]
            for column in range(row + 1, len(lower))
        )
        solution[row] = residual / lower[row][row]
    return tuple(solution)


class LinUCB:
    """Linear Upper Confidence Bound policy with explicit reward updates.

    Args:
        n_features: Context dimension; defaults to the project's 15 features.
        alpha: Nonnegative exploration coefficient, chosen by experiment config.

    The model starts with ``A = I`` and ``b = 0``. Cholesky solves are used
    instead of explicitly forming ``A^-1``. The score remains exactly the
    LinUCB expression: ``x.T @ theta + alpha * sqrt(x.T @ A^-1 @ x)``.
    """

    def __init__(self, n_features: int = DEFAULT_N_FEATURES, *, alpha: float) -> None:
        if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features <= 0:
            raise ValueError("n_features must be a positive integer")
        if isinstance(alpha, bool) or not isinstance(alpha, Real):
            raise ValueError("alpha must be a finite numeric value greater than or equal to 0")
        alpha_value = float(alpha)
        if not math.isfinite(alpha_value) or alpha_value < 0.0:
            raise ValueError("alpha must be a finite numeric value greater than or equal to 0")

        self._n_features = n_features
        self._alpha = alpha_value
        self._a = tuple(
            tuple(1.0 if row == column else 0.0 for column in range(n_features))
            for row in range(n_features)
        )
        self._b = (0.0,) * n_features

    @property
    def n_features(self) -> int:
        """Context dimension used by this model."""
        return self._n_features

    @property
    def alpha(self) -> float:
        """Configured nonnegative exploration coefficient."""
        return self._alpha

    @property
    def A(self) -> tuple[tuple[float, ...], ...]:
        """Read-only snapshot of the current design matrix."""
        return self._a

    @property
    def b(self) -> tuple[float, ...]:
        """Read-only snapshot of the current response vector."""
        return self._b

    @property
    def theta(self) -> tuple[float, ...]:
        """Current estimate ``A^-1 b``, calculated using Cholesky solves."""
        lower = _cholesky(self._a)
        intermediate = _solve_lower(lower, self._b)
        return _solve_upper_from_lower(lower, intermediate)

    def _score_vectors(
        self,
        vectors: tuple[tuple[float, ...], ...],
        candidate_ids: Optional[tuple[Hashable, ...]] = None,
    ) -> tuple[LinUCBScore, ...]:
        if not vectors:
            return ()
        lower = _cholesky(self._a)
        theta = _solve_upper_from_lower(lower, _solve_lower(lower, self._b))
        scores: list[LinUCBScore] = []
        for index, vector in enumerate(vectors):
            mean = math.fsum(value * weight for value, weight in zip(vector, theta))
            whitened = _solve_lower(lower, vector)
            quadratic_form = math.fsum(value * value for value in whitened)
            if not math.isfinite(mean) or not math.isfinite(quadratic_form):
                raise ArithmeticError("LinUCB score overflowed; check context magnitude")
            exploration = self._alpha * math.sqrt(max(0.0, quadratic_form))
            score = mean + exploration
            if not math.isfinite(score):
                raise ArithmeticError("LinUCB score is not finite")
            scores.append(
                LinUCBScore(
                    index=index,
                    candidate_id=(candidate_ids[index] if candidate_ids is not None else None),
                    mean=mean,
                    exploration=exploration,
                    score=score,
                )
            )
        return tuple(scores)

    def score_components(self, context: Iterable[Real]) -> LinUCBScore:
        """Return mean, exploration bonus, and total score for one context."""
        vector = _validated_vector(context, self._n_features)
        return self._score_vectors((vector,))[0]

    def score(self, context: Iterable[Real]) -> float:
        """Return the LinUCB score for a single candidate context."""
        return self.score_components(context).score

    def score_batch(
        self,
        contexts: Iterable[Iterable[Real]],
        candidate_ids: Optional[Iterable[Hashable]] = None,
    ) -> tuple[LinUCBScore, ...]:
        """Score a batch; optional IDs are preserved in each result.

        Equal scores remain ordered by input position. This deterministic
        tie-break applies to ranking and does not modify model state.
        """
        vectors = tuple(
            _validated_vector(context, self._n_features) for context in contexts
        )
        ids = tuple(candidate_ids) if candidate_ids is not None else None
        if ids is not None and len(ids) != len(vectors):
            raise ValueError("candidate_ids length must match contexts length")
        return self._score_vectors(vectors, ids)

    def select_top1(
        self,
        contexts: Iterable[Iterable[Real]],
        candidate_ids: Optional[Iterable[Hashable]] = None,
    ) -> LinUCBScore:
        """Select the single highest-scoring action without updating the model."""
        scores = self.score_batch(contexts, candidate_ids)
        if not scores:
            raise ValueError("cannot select Top-1 from an empty candidate set")
        return max(scores, key=lambda item: (item.score, -item.index))

    def top_k(
        self,
        contexts: Iterable[Iterable[Real]],
        k: int,
        candidate_ids: Optional[Iterable[Hashable]] = None,
    ) -> tuple[LinUCBScore, ...]:
        """Return ranked Top-K for offline evaluation only; performs no updates."""
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        scores = self.score_batch(contexts, candidate_ids)
        return tuple(sorted(scores, key=lambda item: (-item.score, item.index))[:k])

    def update(self, context: Iterable[Real], reward: Real) -> None:
        """Apply one explicit rank-one update for the action that was played.

        The caller decides when feedback is available and should pass only the
        selected Top-1 context. This method does not model delayed feedback.
        """
        vector = _validated_vector(context, self._n_features)
        if isinstance(reward, bool) or not isinstance(reward, Real):
            raise ValueError("reward must be a finite numeric value")
        reward_value = float(reward)
        if not math.isfinite(reward_value):
            raise ValueError("reward must be a finite numeric value")

        new_a = tuple(
            tuple(self._a[row][column] + vector[row] * vector[column]
                  for column in range(self._n_features))
            for row in range(self._n_features)
        )
        new_b = tuple(
            self._b[index] + reward_value * vector[index]
            for index in range(self._n_features)
        )
        if not all(math.isfinite(value) for row in new_a for value in row):
            raise ArithmeticError("LinUCB matrix update overflowed")
        if not all(math.isfinite(value) for value in new_b):
            raise ArithmeticError("LinUCB response update overflowed")
        self._a = new_a
        self._b = new_b
