"""Unit tests for :class:`fastar.FeatureSpec`."""

from __future__ import annotations

import pytest

from fastar import FeatureSpec


def test_basic_construction():
    spec = FeatureSpec(
        columns=("a", "b", "c"),
        continuous=("a", "b"),
        immutable=("c",),
        monotonic_increasing=("a",),
    )
    assert spec.n_features == 3
    assert spec.columns == ("a", "b", "c")
    assert spec.categorical == ("c",)
    assert spec.index("b") == 1
    assert spec.is_immutable(2)
    assert not spec.is_immutable(0)
    assert spec.is_monotonic_increasing(0)
    assert not spec.is_monotonic_increasing(1)


def test_step_size_default():
    spec = FeatureSpec(columns=("a",))
    assert spec.step_size == 0.05


def test_rejects_duplicate_columns():
    with pytest.raises(ValueError, match="duplicates"):
        FeatureSpec(columns=("a", "a", "b"))


def test_rejects_unknown_continuous():
    with pytest.raises(ValueError, match="continuous"):
        FeatureSpec(columns=("a", "b"), continuous=("c",))


def test_rejects_unknown_immutable():
    with pytest.raises(ValueError, match="immutable"):
        FeatureSpec(columns=("a", "b"), immutable=("c",))


def test_rejects_unknown_monotonic():
    with pytest.raises(ValueError, match="monotonic_increasing"):
        FeatureSpec(columns=("a",), monotonic_increasing=("b",))


def test_rejects_immutable_and_monotonic_overlap():
    with pytest.raises(ValueError, match="both immutable and monotonic"):
        FeatureSpec(
            columns=("a", "b"),
            immutable=("a",),
            monotonic_increasing=("a",),
        )


def test_correlated_validates_column_names():
    # Valid: both columns exist.
    spec = FeatureSpec(columns=("a", "b"), correlated=(("a", "b", 0.05),))
    assert spec.correlated == (("a", "b", 0.05),)

    with pytest.raises(ValueError, match="correlated cause"):
        FeatureSpec(columns=("a", "b"), correlated=(("c", "a", 0.05),))

    with pytest.raises(ValueError, match="correlated effect"):
        FeatureSpec(columns=("a", "b"), correlated=(("a", "c", 0.05),))


def test_step_size_must_be_positive():
    with pytest.raises(ValueError, match="step_size"):
        FeatureSpec(columns=("a",), step_size=0.0)
    with pytest.raises(ValueError, match="step_size"):
        FeatureSpec(columns=("a",), step_size=-0.1)


def test_continuous_and_immutable_can_overlap():
    """A feature can be numerical AND forbidden from changing.

    German Credit's ``Number-of-people-being-lible`` is exactly this case.
    """
    spec = FeatureSpec(
        columns=("a", "b"),
        continuous=("a",),
        immutable=("a",),
    )
    assert spec.is_immutable(0)
    assert "a" in spec.continuous
