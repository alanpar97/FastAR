"""Gymnasium environment used to train a FastAR agent.

The agent interacts with a black-box classifier and learns to edit an
undesirable input until the classifier flips to the target class. The
observation is the current feature vector (assumed to live in
``[-1, 1]^n``); the action chooses a feature and a direction
(``feature_idx * 2 + decrease``). The reward is the classifier's probability
of the target class minus a KNN-based manifold penalty, with a large bonus
on termination.

Users rarely instantiate ``CounterfactualEnv`` directly — the high-level
:class:`fastar.FastAR` class builds one internally. It is exposed as a
public class because downstream users may want to wrap it with their own
Gymnasium wrappers (monitoring, logging, curriculum, etc.).
"""

from __future__ import annotations

from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn.neighbors import NearestNeighbors

from .feature_spec import FeatureSpec


class _ProbabilisticClassifier(Protocol):
    """Structural type for any scikit-learn-like classifier."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class CounterfactualEnv(gym.Env):
    """A Gymnasium environment for training FastAR's recourse policy.

    Parameters
    ----------
    classifier:
        Any object exposing ``predict_proba(X)`` returning an array of shape
        ``(n, n_classes)``. The classifier is consulted to compute the
        reward; it is never modified by the environment.
    training_data:
        A 2D array of shape ``(n_samples, n_features)`` already scaled to
        ``[-1, 1]``. Used both as the pool of starting states during
        training and as the reference set for the manifold-distance
        reward term.
    feature_spec:
        A :class:`FeatureSpec` describing the feature layout and
        per-column constraints.
    dist_lambda:
        Coefficient of the manifold-distance penalty in the reward. Set to
        ``0`` to disable the penalty.
    target_class:
        The class label FastAR is trying to reach. Defaults to ``1``.
    max_episode_steps:
        Maximum number of actions per episode before truncation.
    invalid_action_reward:
        Reward returned when the agent attempts a forbidden action
        (touching an immutable feature, decreasing a monotonic feature,
        or stepping out of bounds). Defaults to ``-10``.
    success_bonus:
        Reward bonus on successful termination (classifier flips to
        ``target_class``). Defaults to ``100``.
    n_neighbors_reward:
        Number of neighbors averaged when computing the manifold-distance
        penalty. Defaults to ``1`` (the paper setting).
    n_neighbors_knn_fit:
        Number of neighbors used when fitting the underlying
        :class:`NearestNeighbors` index. Defaults to ``5``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        classifier: _ProbabilisticClassifier,
        training_data: np.ndarray,
        feature_spec: FeatureSpec,
        *,
        dist_lambda: float = 0.1,
        target_class: int = 1,
        max_episode_steps: int = 50,
        invalid_action_reward: float = -10.0,
        success_bonus: float = 100.0,
        n_neighbors_reward: int = 1,
        n_neighbors_knn_fit: int = 5,
    ) -> None:
        super().__init__()

        training_data = np.asarray(training_data, dtype=np.float32)
        if training_data.ndim != 2:
            raise ValueError(
                f"training_data must be 2D, got shape {training_data.shape}"
            )
        if training_data.shape[1] != feature_spec.n_features:
            raise ValueError(
                f"training_data has {training_data.shape[1]} features, "
                f"feature_spec has {feature_spec.n_features}"
            )

        self.classifier = classifier
        self.training_data = training_data
        self.feature_spec = feature_spec
        self.dist_lambda = float(dist_lambda)
        self.target_class = int(target_class)
        self.max_episode_steps = int(max_episode_steps)
        self.invalid_action_reward = float(invalid_action_reward)
        self.success_bonus = float(success_bonus)
        self._n_neighbors_reward = int(n_neighbors_reward)

        n_features = feature_spec.n_features
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_features,), dtype=np.float32
        )
        # For every feature the agent can increase (even action) or decrease
        # (odd action) by ``feature_spec.step_size``.
        self.action_space = spaces.Discrete(2 * n_features)

        self._knn = NearestNeighbors(
            n_neighbors=max(n_neighbors_reward, n_neighbors_knn_fit), p=1
        )
        self._knn.fit(training_data)

        self._state: np.ndarray = np.zeros(n_features, dtype=np.float32)
        self._elapsed_steps: int = 0
        # ``_pinned_target`` is set by ``reset(options={"target": x})`` for
        # evaluation; during training it is ``None`` and reset samples
        # uniformly from ``training_data``.
        self._pinned_target: np.ndarray | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._elapsed_steps = 0

        target = None
        if options is not None:
            target = options.get("target")
        if target is None:
            target = self._pinned_target

        if target is not None:
            target = np.asarray(target, dtype=np.float32).reshape(-1)
            if target.shape[0] != self.feature_spec.n_features:
                raise ValueError(
                    f"target has {target.shape[0]} features, expected "
                    f"{self.feature_spec.n_features}"
                )
            self._state = target.copy()
        else:
            idx = int(self.np_random.integers(0, self.training_data.shape[0]))
            self._state = self.training_data[idx].copy()

        return self._state.copy(), {}

    def step(
        self, action: int | np.integer
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action} for {self.action_space}")

        feature_idx = action // 2
        decrease = bool(action % 2)
        self._elapsed_steps += 1

        info: dict[str, Any] = {}

        # Immutable features cannot be touched at all.
        if self.feature_spec.is_immutable(feature_idx):
            return self._finalize_step(
                self._state, self.invalid_action_reward, terminated=False, info=info
            )

        # Monotonic-increasing features cannot be decreased.
        if decrease and self.feature_spec.is_monotonic_increasing(feature_idx):
            return self._finalize_step(
                self._state, self.invalid_action_reward, terminated=False, info=info
            )

        delta = (
            -self.feature_spec.step_size if decrease else self.feature_spec.step_size
        )
        next_state = self._state.copy()
        next_state[feature_idx] = next_state[feature_idx] + delta

        # Apply correlated-feature couplings: when ``cause`` increases,
        # ``effect`` is bumped by ``delta``.
        if not decrease:
            for (
                cause_idx,
                effect_idx,
                coupling_delta,
            ) in self.feature_spec._correlated_idx:
                if cause_idx == feature_idx:
                    next_state[effect_idx] = next_state[effect_idx] + coupling_delta

        # Bounds check: stepping out of ``[-1, 1]`` is treated as an
        # invalid action and the state is not updated.
        if np.any(next_state < -1.0) or np.any(next_state > 1.0):
            return self._finalize_step(
                self._state, self.invalid_action_reward, terminated=False, info=info
            )

        # Valid action: compute classifier-based reward.
        probs = self.classifier.predict_proba(next_state.reshape(1, -1))[0]
        target_prob = float(probs[self.target_class])
        manifold_penalty = self.dist_lambda * self._manifold_distance(next_state)
        terminated = target_prob >= 0.5
        reward = (self.success_bonus if terminated else target_prob) - manifold_penalty

        self._state = next_state
        return self._finalize_step(
            self._state, reward, terminated=terminated, info=info
        )

    def render(self) -> None:  # pragma: no cover - trivial
        print(f"state={self._state}")

    def _finalize_step(
        self,
        state: np.ndarray,
        reward: float,
        *,
        terminated: bool,
        info: dict[str, Any],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        truncated = (not terminated) and self._elapsed_steps >= self.max_episode_steps
        return state.copy(), float(reward), bool(terminated), bool(truncated), info

    def _manifold_distance(self, state: np.ndarray) -> float:
        dist, _ = self._knn.kneighbors(
            state.reshape(1, -1),
            n_neighbors=self._n_neighbors_reward,
            return_distance=True,
        )
        return float(np.mean(dist))

    def set_pinned_target(self, target: np.ndarray | None) -> None:
        """Pin subsequent ``reset`` calls to a specific starting state.

        Passing ``None`` clears the pin and restores random sampling from
        ``training_data``. This is used by :meth:`fastar.FastAR.explain` to
        force the evaluation environment to start from a given instance
        without having to plumb ``options`` through Stable-Baselines3's
        ``VecEnv`` machinery.
        """
        if target is None:
            self._pinned_target = None
            return
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        if target.shape[0] != self.feature_spec.n_features:
            raise ValueError(
                f"target has {target.shape[0]} features, expected "
                f"{self.feature_spec.n_features}"
            )
        self._pinned_target = target
