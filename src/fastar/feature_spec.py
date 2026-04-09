"""Feature metadata for a counterfactual recourse problem.

A ``FeatureSpec`` describes the columns of a tabular dataset and the
per-column constraints that FastAR must respect when searching for a
counterfactual: which features are immutable, which can only increase, and
which feature changes imply mechanical changes in other features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class FeatureSpec:
    """Describes the feature layout of a tabular dataset for FastAR.

    Parameters
    ----------
    columns:
        Ordered sequence of column names. Defines the feature index used by
        the environment's action space.
    continuous:
        Names of continuous (numerical) columns. Columns not listed here are
        treated as categorical.
    immutable:
        Names of columns the agent is not allowed to change (e.g. sex, race).
    monotonic_increasing:
        Names of columns the agent may only increase, never decrease
        (e.g. ``age``).
    correlated:
        Tuples of ``(cause, effect, delta)`` describing a mechanical coupling:
        whenever ``cause`` is increased, ``effect`` is also increased by
        ``delta`` (in the same scaled units as the observation space). Use
        this for relationships like "increasing education level implies an
        increase in age".
    step_size:
        Magnitude of a single action step in the normalized ``[-1, 1]``
        feature space. Defaults to ``0.05`` (the value used in the paper).
    """

    columns: tuple[str, ...]
    continuous: tuple[str, ...] = ()
    immutable: tuple[str, ...] = ()
    monotonic_increasing: tuple[str, ...] = ()
    correlated: tuple[tuple[str, str, float], ...] = ()
    step_size: float = 0.05

    # Derived lookup structures (populated in __post_init__).
    _column_index: dict[str, int] = field(init=False, repr=False, compare=False)
    _immutable_idx: frozenset[int] = field(init=False, repr=False, compare=False)
    _monotonic_idx: frozenset[int] = field(init=False, repr=False, compare=False)
    _correlated_idx: tuple[tuple[int, int, float], ...] = field(
        init=False, repr=False, compare=False
    )

    def __init__(
        self,
        columns: Sequence[str],
        continuous: Sequence[str] = (),
        immutable: Sequence[str] = (),
        monotonic_increasing: Sequence[str] = (),
        correlated: Sequence[tuple[str, str, float]] = (),
        step_size: float = 0.05,
    ) -> None:
        columns_t = tuple(columns)
        if len(set(columns_t)) != len(columns_t):
            raise ValueError("FeatureSpec.columns contains duplicates")
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")

        column_index = {name: i for i, name in enumerate(columns_t)}

        def _check_subset(kind: str, names: Sequence[str]) -> tuple[str, ...]:
            unknown = [n for n in names if n not in column_index]
            if unknown:
                raise ValueError(f"{kind} contains unknown columns: {unknown}")
            return tuple(names)

        continuous_t = _check_subset("continuous", continuous)
        immutable_t = _check_subset("immutable", immutable)
        monotonic_t = _check_subset("monotonic_increasing", monotonic_increasing)

        overlap = set(immutable_t) & set(monotonic_t)
        if overlap:
            raise ValueError(
                f"columns cannot be both immutable and monotonic_increasing: {sorted(overlap)}"
            )

        correlated_t: list[tuple[str, str, float]] = []
        correlated_idx_list: list[tuple[int, int, float]] = []
        for entry in correlated:
            cause, effect, delta = entry
            if cause not in column_index:
                raise ValueError(f"correlated cause column unknown: {cause}")
            if effect not in column_index:
                raise ValueError(f"correlated effect column unknown: {effect}")
            correlated_t.append((cause, effect, float(delta)))
            correlated_idx_list.append(
                (column_index[cause], column_index[effect], float(delta))
            )

        # ``frozen=True`` forbids normal assignment, so use object.__setattr__.
        object.__setattr__(self, "columns", columns_t)
        object.__setattr__(self, "continuous", continuous_t)
        object.__setattr__(self, "immutable", immutable_t)
        object.__setattr__(self, "monotonic_increasing", monotonic_t)
        object.__setattr__(self, "correlated", tuple(correlated_t))
        object.__setattr__(self, "step_size", float(step_size))
        object.__setattr__(self, "_column_index", column_index)
        object.__setattr__(
            self, "_immutable_idx", frozenset(column_index[n] for n in immutable_t)
        )
        object.__setattr__(
            self, "_monotonic_idx", frozenset(column_index[n] for n in monotonic_t)
        )
        object.__setattr__(self, "_correlated_idx", tuple(correlated_idx_list))

    @property
    def n_features(self) -> int:
        return len(self.columns)

    @property
    def categorical(self) -> tuple[str, ...]:
        """Columns not listed as continuous."""
        cont = set(self.continuous)
        return tuple(c for c in self.columns if c not in cont)

    def index(self, name: str) -> int:
        """Return the integer index of a column by name."""
        return self._column_index[name]

    def is_immutable(self, feature_idx: int) -> bool:
        return feature_idx in self._immutable_idx

    def is_monotonic_increasing(self, feature_idx: int) -> bool:
        return feature_idx in self._monotonic_idx
