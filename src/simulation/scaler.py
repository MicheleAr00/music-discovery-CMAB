"""Warm-up-only standardization for the 15 raw context features.

The first thirteen features are standardized with population statistics
(``ddof=0``). The final two novelty flags stay binary and unchanged.
"""

from __future__ import annotations

import math
from itertools import chain
from numbers import Real
from typing import Iterable, Iterator, TypeAlias


FEATURE_COUNT = 15
STANDARDIZED_FEATURE_COUNT = 13
ContextVector: TypeAlias = tuple[float, ...]
ContextMatrix: TypeAlias = tuple[ContextVector, ...]
ContextInput: TypeAlias = Iterable[Real] | Iterable[Iterable[Real]]
ContextOutput: TypeAlias = ContextVector | ContextMatrix


def _validate_vector(values: Iterable[Real]) -> ContextVector:
    """Convert and validate one raw context vector."""
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("context values must be numeric") from exc
    if len(vector) != FEATURE_COUNT:
        raise ValueError(
            f"each context must contain exactly {FEATURE_COUNT} features"
        )
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("context values must all be finite")
    if vector[13] not in (0.0, 1.0) or vector[14] not in (0.0, 1.0):
        raise ValueError("the final two novelty features must be binary 0/1")
    return vector


def _context_rows(
    contexts: ContextInput,
) -> tuple[Iterator[ContextVector], bool]:
    """Return validated rows and whether the input was a single vector."""
    try:
        iterator = iter(contexts)
        first = next(iterator)
    except (TypeError, StopIteration) as exc:
        raise ValueError("contexts must contain at least one 15-feature vector") from exc

    if isinstance(first, Real):
        row = _validate_vector(chain((first,), iterator))
        return iter((row,)), True

    rows = (_validate_vector(row) for row in chain((first,), iterator))
    return rows, False


class FeatureScaler:
    """Fit once on warm-up contexts, then transform contexts without refitting.

    ``fit`` accepts either one 15-value context or an iterable of contexts.
    It computes means and population standard deviations for columns 1–13
    using an online float64 Welford calculation. A zero standard deviation is
    replaced by 1, so a constant feature maps to zero at its fitted value and
    no arbitrary epsilon is introduced. The last two columns are novelty flags:
    they are validated as 0/1, assigned mean 0 and scale 1, and left unchanged.

    An instance may be fitted only once. Create a new scaler to fit a different
    warm-up sample; this prevents accidental validation/test refits.
    Inputs and outputs are tuples, so a single vector has logical shape
    ``(15,)`` and a batch has logical shape ``(n, 15)`` without requiring NumPy.
    """

    def __init__(self) -> None:
        self._mean: ContextVector | None = None
        self._scale: ContextVector | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether warm-up statistics have been fitted."""
        return self._mean is not None

    @property
    def mean_(self) -> ContextVector:
        """Fitted per-feature means; unavailable before :meth:`fit`."""
        self._require_fitted()
        assert self._mean is not None
        return self._mean

    @property
    def scale_(self) -> ContextVector:
        """Fitted standard deviations (zero variance replaced by 1)."""
        self._require_fitted()
        assert self._scale is not None
        return self._scale

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted on warm-up data first")

    def fit(self, contexts: ContextInput) -> FeatureScaler:
        """Estimate fixed parameters from warm-up contexts only.

        Uses population standard deviation (``ddof=0``). Calling ``fit`` more
        than once on an instance raises ``RuntimeError``; this makes the
        warm-up-only boundary explicit and prevents later data from changing
        fitted parameters.
        """
        if self.is_fitted:
            raise RuntimeError(
                "FeatureScaler is already fitted; create a new instance to refit"
            )

        rows, _ = _context_rows(contexts)
        count = 0
        means = [0.0] * STANDARDIZED_FEATURE_COUNT
        sums_squared_deviation = [0.0] * STANDARDIZED_FEATURE_COUNT
        for row in rows:
            count += 1
            for index in range(STANDARDIZED_FEATURE_COUNT):
                delta = row[index] - means[index]
                means[index] += delta / count
                delta_after_update = row[index] - means[index]
                sums_squared_deviation[index] += delta * delta_after_update

        if count == 0:
            raise ValueError("fit requires at least one warm-up context")

        scales: list[float] = []
        for sum_squared_deviation in sums_squared_deviation:
            variance = max(0.0, sum_squared_deviation / count)
            standard_deviation = math.sqrt(variance)
            scales.append(standard_deviation if standard_deviation > 0.0 else 1.0)

        self._mean = tuple(means + [0.0, 0.0])
        self._scale = tuple(scales + [1.0, 1.0])
        return self

    def transform(self, contexts: ContextInput) -> ContextOutput:
        """Transform one context or a batch using fixed warm-up parameters."""
        self._require_fitted()
        assert self._mean is not None
        assert self._scale is not None
        rows, is_single = _context_rows(contexts)
        transformed = tuple(
            tuple(
                (value - self._mean[index]) / self._scale[index]
                if index < STANDARDIZED_FEATURE_COUNT
                else value
                for index, value in enumerate(row)
            )
            for row in rows
        )
        if is_single:
            return transformed[0]
        return transformed

    def fit_transform(self, contexts: ContextInput) -> ContextOutput:
        """Fit on the supplied warm-up data and return its transformed form."""
        rows, is_single = _context_rows(contexts)
        materialized = tuple(rows)
        self.fit(materialized)
        return self.transform(materialized[0] if is_single else materialized)
