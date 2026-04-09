"""Unit tests for :class:`fastar.CounterfactualEnv`."""

from __future__ import annotations

import numpy as np
import pytest

from fastar import CounterfactualEnv, FeatureSpec

from .fixtures.toy import make_blobs, make_classifier


def _make_env(max_episode_steps: int = 20):
    X, y = make_blobs()
    classifier = make_classifier(X, y)
    spec = FeatureSpec(columns=("x0", "x1"), continuous=("x0", "x1"))
    return (
        CounterfactualEnv(
            classifier=classifier,
            training_data=X,
            feature_spec=spec,
            dist_lambda=0.0,
            max_episode_steps=max_episode_steps,
        ),
        X,
        classifier,
    )


def test_spaces_match_feature_spec():
    env, _, _ = _make_env()
    assert env.observation_space.shape == (2,)
    assert env.action_space.n == 4  # 2 features * 2 directions


def test_reset_samples_from_training_data():
    env, X, _ = _make_env()
    obs, info = env.reset(seed=42)
    assert obs.shape == (2,)
    assert info == {}
    # The sampled observation must equal one of the training rows.
    assert np.any(np.all(np.isclose(X, obs), axis=1))


def test_reset_with_explicit_target():
    env, _, _ = _make_env()
    target = np.array([-0.3, -0.3], dtype=np.float32)
    obs, _ = env.reset(options={"target": target})
    assert np.allclose(obs, target)


def test_pinned_target_persists_across_resets():
    env, _, _ = _make_env()
    target = np.array([-0.2, -0.2], dtype=np.float32)
    env.set_pinned_target(target)
    obs_a, _ = env.reset()
    obs_b, _ = env.reset()
    assert np.allclose(obs_a, target)
    assert np.allclose(obs_b, target)
    env.set_pinned_target(None)


def test_step_increment_and_decrement():
    env, _, _ = _make_env()
    env.reset(options={"target": np.array([0.0, 0.0], dtype=np.float32)})
    # action 0 = feature 0 increase
    obs, _, _, _, _ = env.step(0)
    assert obs[0] == pytest.approx(0.05)
    # action 1 = feature 0 decrease
    obs, _, _, _, _ = env.step(1)
    assert obs[0] == pytest.approx(0.0, abs=1e-6)
    # action 2 = feature 1 increase
    obs, _, _, _, _ = env.step(2)
    assert obs[1] == pytest.approx(0.05)


def test_immutable_feature_is_a_noop():
    X, y = make_blobs()
    classifier = make_classifier(X, y)
    spec = FeatureSpec(columns=("x0", "x1"), continuous=("x0",), immutable=("x1",))
    env = CounterfactualEnv(
        classifier=classifier,
        training_data=X,
        feature_spec=spec,
        dist_lambda=0.0,
    )
    start = np.array([0.0, 0.0], dtype=np.float32)
    env.reset(options={"target": start})
    # Action 2 = try to increase immutable feature x1.
    obs, reward, terminated, _, _ = env.step(2)
    assert np.allclose(obs, start)
    assert reward == env.invalid_action_reward
    assert not terminated


def test_monotonic_decrease_is_a_noop():
    X, y = make_blobs()
    classifier = make_classifier(X, y)
    spec = FeatureSpec(
        columns=("x0", "x1"),
        continuous=("x0", "x1"),
        monotonic_increasing=("x0",),
    )
    env = CounterfactualEnv(
        classifier=classifier,
        training_data=X,
        feature_spec=spec,
        dist_lambda=0.0,
    )
    start = np.array([0.0, 0.0], dtype=np.float32)
    env.reset(options={"target": start})
    # Action 1 = decrease x0, which is forbidden.
    obs, reward, terminated, _, _ = env.step(1)
    assert np.allclose(obs, start)
    assert reward == env.invalid_action_reward
    # Increase is still allowed.
    obs, _, _, _, _ = env.step(0)
    assert obs[0] == pytest.approx(0.05)


def test_correlated_feature_bump_on_increase():
    X, y = make_blobs()
    classifier = make_classifier(X, y)
    spec = FeatureSpec(
        columns=("x0", "x1"),
        continuous=("x0", "x1"),
        correlated=(("x0", "x1", 0.10),),
    )
    env = CounterfactualEnv(
        classifier=classifier,
        training_data=X,
        feature_spec=spec,
        dist_lambda=0.0,
    )
    env.reset(options={"target": np.array([0.0, 0.0], dtype=np.float32)})
    obs, _, _, _, _ = env.step(0)  # increase x0 -> x1 also bumps
    assert obs[0] == pytest.approx(0.05)
    assert obs[1] == pytest.approx(0.10)


def test_correlated_feature_no_bump_on_decrease():
    X, y = make_blobs()
    classifier = make_classifier(X, y)
    spec = FeatureSpec(
        columns=("x0", "x1"),
        continuous=("x0", "x1"),
        correlated=(("x0", "x1", 0.10),),
    )
    env = CounterfactualEnv(
        classifier=classifier,
        training_data=X,
        feature_spec=spec,
        dist_lambda=0.0,
    )
    env.reset(options={"target": np.array([0.3, 0.3], dtype=np.float32)})
    obs, _, _, _, _ = env.step(1)  # decrease x0 -> no bump on x1
    assert obs[0] == pytest.approx(0.25)
    assert obs[1] == pytest.approx(0.30)


def test_out_of_bounds_step_is_rejected():
    env, _, _ = _make_env()
    env.reset(options={"target": np.array([1.0, 0.0], dtype=np.float32)})
    # action 0 = try to increase x0 past +1.0
    obs, reward, _, _, _ = env.step(0)
    assert obs[0] == pytest.approx(1.0)
    assert reward == env.invalid_action_reward


def test_truncation_after_max_episode_steps():
    env, _, _ = _make_env(max_episode_steps=3)
    env.reset(options={"target": np.array([-0.9, -0.9], dtype=np.float32)})
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(0)
        assert not truncated
        assert not terminated
    _, _, terminated, truncated, _ = env.step(0)
    assert truncated or terminated


def test_reaching_target_class_terminates():
    X, y = make_blobs()
    classifier = make_classifier(X, y)
    spec = FeatureSpec(columns=("x0", "x1"), continuous=("x0", "x1"))
    env = CounterfactualEnv(
        classifier=classifier,
        training_data=X,
        feature_spec=spec,
        dist_lambda=0.0,
        max_episode_steps=200,
    )
    # Start on the class-0 side; repeatedly step towards class 1 by
    # increasing both features.
    env.reset(options={"target": np.array([-0.9, -0.9], dtype=np.float32)})
    terminated = False
    for _ in range(200):
        _, reward, terminated, _, _ = env.step(0)  # increase x0
        if terminated:
            break
        _, reward, terminated, _, _ = env.step(2)  # increase x1
        if terminated:
            break
    assert terminated
    assert reward >= env.success_bonus - 1.0  # success bonus minus small penalty


def test_rejects_shape_mismatch_in_training_data():
    _, y = make_blobs()
    X_wrong = np.zeros((len(y), 3), dtype=np.float32)
    classifier = make_classifier(np.zeros((len(y), 2)), y)
    spec = FeatureSpec(columns=("x0", "x1"))
    with pytest.raises(ValueError, match="feature_spec"):
        CounterfactualEnv(
            classifier=classifier, training_data=X_wrong, feature_spec=spec
        )
