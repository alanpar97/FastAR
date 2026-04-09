"""Smoke test for the full :class:`fastar.FastAR` fit/explain loop.

Uses a tiny training budget — the point of this test is to catch API
breakage and integration bugs, not to verify that FastAR produces high
quality counterfactuals.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastar import Explanation, FastAR

from .fixtures.toy import make_blobs, make_classifier, make_spec


@pytest.fixture(scope="module")
def trained_fastar():
    X, y = make_blobs(n_per_class=60, seed=0)
    classifier = make_classifier(X, y)
    spec = make_spec()
    fastar = FastAR(
        classifier=classifier,
        feature_spec=spec,
        dist_lambda=0.0,
        max_episode_steps=30,
        seed=0,
    )
    fastar.fit(X, total_timesteps=2_000, ppo_kwargs={"verbose": 0})
    return fastar, classifier, X


def test_explain_returns_explanation_for_class_0_instance(trained_fastar):
    fastar, classifier, X = trained_fastar
    class0_mask = classifier.predict(X) == 0
    assert class0_mask.any()
    x = X[class0_mask][0]

    explanation = fastar.explain(x)

    assert isinstance(explanation, Explanation)
    assert explanation.original.shape == (2,)
    assert explanation.counterfactual.shape == (2,)
    assert explanation.trajectory.ndim == 2
    assert explanation.trajectory.shape[1] == 2
    assert explanation.trajectory.shape[0] == explanation.num_steps + 1
    assert np.allclose(explanation.trajectory[0], explanation.original)
    assert np.allclose(explanation.trajectory[-1], explanation.counterfactual)


def test_explain_batch(trained_fastar):
    fastar, classifier, X = trained_fastar
    class0 = X[classifier.predict(X) == 0][:3]
    explanations = fastar.explain_batch(class0)
    assert len(explanations) == 3
    for exp in explanations:
        assert isinstance(exp, Explanation)


def test_explain_before_fit_raises():
    fastar = FastAR(classifier=None, feature_spec=make_spec())  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    with pytest.raises(RuntimeError, match="before fit"):
        fastar.explain(np.zeros(2, dtype=np.float32))


def test_model_property_before_fit_raises():
    fastar = FastAR(classifier=None, feature_spec=make_spec())  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    with pytest.raises(RuntimeError, match="before fit"):
        _ = fastar.model


def test_save_and_load_policy(tmp_path, trained_fastar):
    fastar, classifier, X = trained_fastar
    save_path = tmp_path / "policy.zip"
    fastar.save(save_path)
    assert save_path.exists()

    fresh = FastAR(
        classifier=classifier,
        feature_spec=make_spec(),
        dist_lambda=0.0,
        max_episode_steps=30,
        seed=0,
    )
    fresh.load_policy(save_path, X)

    class0 = X[classifier.predict(X) == 0][0]
    original_explanation = fastar.explain(class0)
    loaded_explanation = fresh.explain(class0)
    assert np.allclose(
        original_explanation.counterfactual, loaded_explanation.counterfactual
    )
