"""End-to-end FastAR pipeline on the German Credit dataset.

Demonstrates the full recipe a typical user follows:

    1. Load a tabular dataset with pandas.
    2. Split, scale to [-1, 1], train any scikit-learn classifier with
       ``predict_proba``.
    3. Encode domain knowledge in a :class:`fastar.FeatureSpec`:
       which columns are continuous, which are immutable (sex, marital
       status), which can only increase (age, job level).
    4. Fit :class:`fastar.FastAR` on the scaled training data.
    5. Pick an instance the classifier predicted as the undesirable class
       and ask FastAR for a counterfactual.

The hyperparameters used here are intentionally *smaller* than the paper
so the example runs in a few minutes on a laptop. If you want to reproduce
the paper numbers, see the original repository.

Run from the repository root with::

    python examples/german_credit.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

from fastar import FastAR, FeatureSpec

# --------------------------------------------------------------------- Config

DATA_PATH = Path(__file__).parent / "datasets" / "german_redone.csv"

# Reproducibility.
RANDOM_STATE = 26
FASTAR_SEED = 0

# Training budget for the example.
TOTAL_TIMESTEPS = 200_000

# Domain knowledge for German Credit
# adapting FastAR to a new dataset is a matter of editing the ``FeatureSpec`` below.
CONTINUOUS_FEATURES: tuple[str, ...] = (
    "Months",
    "Credit-amount",
    "Insatllment-rate",
    "Present-residence-since",
    "age",
    "Number-of-existing-credits",
    "Number-of-people-being-lible",
)
IMMUTABLE_FEATURES: tuple[str, ...] = (
    "Personal-status",
    "Number-of-people-being-lible",
    "Foreign-worker",
    "Purpose",
)
MONOTONIC_INCREASING_FEATURES: tuple[str, ...] = ("age", "Job")
CORRELATED_FEATURES: tuple[tuple[str, str, float], ...] = ()

def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    y = df["target"]
    X = df.drop(columns=["target"])
    return X, y


def train_classifier(X_train_scaled: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """Train a small MLP ."""
    classifier = MLPClassifier(
        solver="lbfgs",
        alpha=1e-5,
        hidden_layer_sizes=(5, 3),
        max_iter=10000,
        random_state=RANDOM_STATE,
    )
    classifier.fit(X_train_scaled, y_train)
    return classifier


def pick_undesirable_instance(
    classifier: MLPClassifier, X_scaled: np.ndarray
) -> np.ndarray:
    """Return the first row the classifier predicts as class 0."""
    preds = classifier.predict(X_scaled)
    mask = preds == 0
    if not mask.any():
        raise RuntimeError("classifier did not predict class 0 for any instance")
    return X_scaled[mask][0]


def describe_explanation(
    explanation,
    columns: list[str],
    classifier: MLPClassifier,
) -> None:
    original = explanation.original
    counterfactual = explanation.counterfactual
    diff = counterfactual - original
    changed = np.flatnonzero(np.abs(diff) > 1e-6)

    prob_before = classifier.predict_proba(original.reshape(1, -1))[0, 1]
    prob_after = classifier.predict_proba(counterfactual.reshape(1, -1))[0, 1]

    print()
    print("=" * 64)
    print("FastAR explanation")
    print("=" * 64)
    print(f"success      : {explanation.success}")
    print(f"num_steps    : {explanation.num_steps}")
    print(f"P(class 1)   : {prob_before:.3f}  ->  {prob_after:.3f}")
    print()
    print("changed features (values are in the scaled [-1, 1] space):")
    if len(changed) == 0:
        print("  (none)")
    else:
        name_width = max(len(columns[i]) for i in changed)
        for i in changed:
            name = columns[i].ljust(name_width)
            print(
                f"  {name}  {original[i]:+.3f}  ->  {counterfactual[i]:+.3f}  "
                f"(delta {diff[i]:+.3f})"
            )


def main() -> None:
    X, y = load_dataset()
    columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(X_train)
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    classifier = train_classifier(X_train_scaled, y_train.to_numpy())
    print(f"classifier test accuracy: {classifier.score(X_test_scaled, y_test):.3f}")

    spec = FeatureSpec(
        columns=columns,
        continuous=CONTINUOUS_FEATURES,
        immutable=IMMUTABLE_FEATURES,
        monotonic_increasing=MONOTONIC_INCREASING_FEATURES,
        correlated=CORRELATED_FEATURES,
    )

    fastar = FastAR(
        classifier=classifier,
        feature_spec=spec,
        dist_lambda=0.1,
        max_episode_steps=50,
        seed=FASTAR_SEED,
    )
    print(f"training FastAR for {TOTAL_TIMESTEPS:,} timesteps...")
    fastar.fit(X_train_scaled, total_timesteps=TOTAL_TIMESTEPS)

    x = pick_undesirable_instance(classifier, X_test_scaled)
    explanation = fastar.explain(x)
    describe_explanation(explanation, columns, classifier)


if __name__ == "__main__":
    main()
