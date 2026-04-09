"""Tiny synthetic classifier + dataset used throughout the test suite.

Deliberately two-dimensional and linearly separable so every test finishes
in well under a second without touching the real German Credit CSV.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from fastar import FeatureSpec


def make_blobs(n_per_class: int = 80, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return two Gaussian blobs in ``[-1, 1]^2``."""
    rng = np.random.default_rng(seed)
    class0 = rng.normal(loc=-0.5, scale=0.12, size=(n_per_class, 2))
    class1 = rng.normal(loc=+0.5, scale=0.12, size=(n_per_class, 2))
    X = np.clip(np.vstack([class0, class1]), -1.0, 1.0).astype(np.float32)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)
    return X, y


def make_classifier(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression().fit(X, y)


def make_spec() -> FeatureSpec:
    """A two-feature spec with no constraints."""
    return FeatureSpec(columns=("x0", "x1"), continuous=("x0", "x1"))
