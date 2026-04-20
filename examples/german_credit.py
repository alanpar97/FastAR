"""End-to-end FastAR pipeline on the German Credit dataset.

Demonstrates the full recipe a typical user follows:

    1. Fetch a tabular dataset (OpenML ``credit-g``, id 31) with no
       bundled CSV in the repository.
    2. Encode ordinal and nominal categoricals to integers, split,
       scale to [-1, 1], and train a small scikit-learn MLP with
       ``predict_proba``.
    3. Encode domain knowledge in a :class:`fastar.FeatureSpec`:
       which columns are continuous, which are immutable (sex/marital
       status, foreign-worker flag, loan purpose), which can only
       increase (age, employment tenure), and which features are
       mechanically coupled (longer employment implies higher age,
       richer credit history implies higher age).
    4. Fit :class:`fastar.FastAR` on the scaled training data.
    5. Pick an instance the classifier predicted as the undesirable
       class ("bad" credit risk) and ask FastAR for a counterfactual
       that would flip it to "good".

Run from the repository root with::

    python examples/german_credit.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

from fastar import Explanation, FastAR, FeatureSpec

# Reproducibility.
RANDOM_STATE = 26
FASTAR_SEED = 0

TOTAL_TIMESTEPS = 300_000

ORDINAL_ENCODINGS: dict[str, tuple[str, ...]] = {
    "checking_status": ("no checking", "<0", "0<=X<200", ">=200"),
    "credit_history": (
        "no credits/all paid",
        "all paid",
        "existing paid",
        "delayed previously",
        "critical/other existing credit",
    ),
    "savings_status": (
        "no known savings",
        "<100",
        "100<=X<500",
        "500<=X<1000",
        ">=1000",
    ),
    "employment": ("unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"),
    "job": (
        "unemp/unskilled non res",
        "unskilled resident",
        "skilled",
        "high qualif/self emp/mgmt",
    ),
}

NOMINAL_CATEGORICALS: tuple[str, ...] = (
    "purpose",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "own_telephone",
    "foreign_worker",
)

# Domain knowledge for German Credit
# Adapting FastAR to a new dataset is a matter of editing the FeatureSpec below.
CONTINUOUS_FEATURES: tuple[str, ...] = (
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
)

# Sex/marital status, loan purpose, and foreign-worker status are facts
# about the applicant that should not be suggested as recourse actions.
IMMUTABLE_FEATURES: tuple[str, ...] = (
    "personal_status",
    "foreign_worker",
    "purpose",
)

# Age only goes up; so does employment tenure if the applicant stays employed.
MONOTONIC_INCREASING_FEATURES: tuple[str, ...] = ("age", "employment")

# Mechanical couplings in scaled ([-1, 1]) space:
#   - Accumulating employment tenure implies the applicant has aged.
#   - Accumulating credit history likewise implies the applicant has aged.
# The 0.054 delta matches the paper's convention (~1.5 years on the
# 19..75 age range scaled to [-1, 1]).
CORRELATED_FEATURES: tuple[tuple[str, str, float], ...] = (
    ("employment", "age", 0.054),
    ("credit_history", "age", 0.054),
)

def load_dataset() -> tuple[pd.DataFrame, pd.Series, dict[str, tuple[str, ...]]]:
    """Fetch credit-g (OpenML id 31) and integer-encode the categoricals.

    Returns the encoded features, the binary target ``(1 = good risk,
    0 = bad risk)``, and a mapping from column name to the ordered list
    of category labels used for the encoding. The mapping is used later
    to format explanations back into human-readable category names.
    """
    ds = fetch_openml(data_id=31, as_frame=True)
    df = ds.frame.copy()

    y = (df["class"] == "good").astype(int)
    df = df.drop(columns=["class"])

    label_maps: dict[str, tuple[str, ...]] = {}

    for col, order in ORDINAL_ENCODINGS.items():
        cat = pd.Categorical(df[col], categories=order, ordered=True)
        if cat.isna().any():
            unknown = sorted(set(df[col]) - set(order))
            raise RuntimeError(f"unexpected values in {col}: {unknown}")
        df[col] = cat.codes.astype(int)
        label_maps[col] = order

    for col in NOMINAL_CATEGORICALS:
        cat = pd.Categorical(df[col])
        df[col] = cat.codes.astype(int)
        label_maps[col] = tuple(cat.categories)

    return df.astype(float), y, label_maps


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
    """Return an undesirable instance.

    Picking an instance with ``P(good) < 0.5`` but close to ``0.5`` makes
    the example reliably find a counterfactual within a reasonable step
    budget. Instances deep inside the "bad" region can be
    unreachable within ``max_episode_steps``.
    """
    probs = classifier.predict_proba(X_scaled)[:, 1]
    mask = probs < 0.5
    if not mask.any():
        raise RuntimeError("classifier did not predict class 0 for any instance")
    candidates = np.where(mask)[0]
    best = candidates[np.argmax(probs[candidates])]
    return X_scaled[best]


def _format_feature_value(
    column: str,
    value: float,
    label_maps: dict[str, tuple[str, ...]],
) -> str:
    """Pretty-print a feature value: category label for categoricals, number otherwise."""
    if column in label_maps:
        idx = int(np.clip(round(value), 0, len(label_maps[column]) - 1))
        return label_maps[column][idx]
    return f"{value:+.2f}"


def describe_explanation(
    explanation: Explanation,
    columns: list[str],
    classifier: MLPClassifier,
    scaler: MinMaxScaler,
    label_maps: dict[str, tuple[str, ...]],
) -> None:
    original = explanation.original
    counterfactual = explanation.counterfactual

    original_unscaled = scaler.inverse_transform(original.reshape(1, -1)).flatten()
    counterfactual_unscaled = scaler.inverse_transform(
        counterfactual.reshape(1, -1)
    ).flatten()

    changed = np.flatnonzero(np.abs(counterfactual - original) > 1e-6)
    prob_before = classifier.predict_proba(original.reshape(1, -1))[0, 1]
    prob_after = classifier.predict_proba(counterfactual.reshape(1, -1))[0, 1]

    print()
    print("=" * 72)
    print("FastAR explanation")
    print("=" * 72)
    print(f"success        : {explanation.success}")
    print(f"num_steps      : {explanation.num_steps}")
    print(f"P(good credit) : {prob_before:.3f}  ->  {prob_after:.3f}")
    print()
    print("changed features:")
    if len(changed) == 0:
        print("  (none)")
        return

    name_width = max(len(columns[i]) for i in changed)
    for i in changed:
        name = columns[i].ljust(name_width)
        before = _format_feature_value(columns[i], original_unscaled[i], label_maps)
        after = _format_feature_value(columns[i], counterfactual_unscaled[i], label_maps)
        print(f"  {name}  {before}  ->  {after}")

def main() -> None:
    print("loading credit-g (OpenML id 31)...")
    X, y, label_maps = load_dataset()
    columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(X_train)
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    classifier = train_classifier(X_train_scaled, y_train.to_numpy())
    print(f"classifier train accuracy: {classifier.score(X_train_scaled, y_train):.3f}")
    print(f"classifier test accuracy:  {classifier.score(X_test_scaled, y_test):.3f}")
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
    describe_explanation(explanation, columns, classifier, scaler, label_maps)

if __name__ == "__main__":
    main()
