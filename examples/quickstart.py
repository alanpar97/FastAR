"""Minimal FastAR example on a 2D synthetic dataset.

Generates two Gaussian blobs in [-1, 1]^2, trains a logistic regression to
separate them, then asks FastAR to produce a counterfactual for a point
that starts on the wrong side of the decision boundary.

Run from the repository root with::

    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from fastar import FastAR, FeatureSpec

RNG_SEED = 0
N_PER_CLASS = 200
TOTAL_TIMESTEPS = 20_000


def make_dataset(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two Gaussian blobs centered at (-0.5, -0.5) and (+0.5, +0.5)."""
    class0 = rng.normal(loc=-0.5, scale=0.15, size=(N_PER_CLASS, 2))
    class1 = rng.normal(loc=+0.5, scale=0.15, size=(N_PER_CLASS, 2))
    X = np.clip(np.vstack([class0, class1]), -1.0, 1.0).astype(np.float32)
    y = np.concatenate([np.zeros(N_PER_CLASS), np.ones(N_PER_CLASS)]).astype(int)
    return X, y


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    X, y = make_dataset(rng)

    classifier = LogisticRegression().fit(X, y)
    train_acc = classifier.score(X, y)
    print(f"classifier train accuracy: {train_acc:.3f}")

    # No constraints at all — every feature is free to move in both
    # directions. This is the simplest possible FeatureSpec.
    spec = FeatureSpec(columns=("x0", "x1"), continuous=("x0", "x1"))

    fastar = FastAR(
        classifier=classifier,
        feature_spec=spec,
        dist_lambda=0.0,  # no manifold penalty for the quickstart
        max_episode_steps=50,
        seed=RNG_SEED,
    )
    fastar.fit(X, total_timesteps=TOTAL_TIMESTEPS)

    # Pick an instance classified as 0 to explain.
    class0_mask = classifier.predict(X) == 0
    x = X[class0_mask][0]
    explanation = fastar.explain(x)

    print()
    print(f"original      : {explanation.original}")
    print(f"counterfactual: {explanation.counterfactual}")
    print(f"success       : {explanation.success}")
    print(f"num_steps     : {explanation.num_steps}")
    print(
        "classifier prob(class 1) "
        f"before: {classifier.predict_proba(x.reshape(1, -1))[0, 1]:.3f} "
        f"after: {classifier.predict_proba(explanation.counterfactual.reshape(1, -1))[0, 1]:.3f}"
    )


if __name__ == "__main__":
    main()
