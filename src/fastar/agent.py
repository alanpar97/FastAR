"""High-level user API for FastAR.

:class:`FastAR` hides the Gymnasium environment and the PPO training loop
behind a small fit/explain interface similar in spirit to scikit-learn's
estimator protocol. A typical use looks like::

    fastar = FastAR(classifier, feature_spec, dist_lambda=0.1)
    fastar.fit(X_train_scaled, total_timesteps=500_000)
    explanation = fastar.explain(x_undesirable)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from .env import CounterfactualEnv, _ProbabilisticClassifier
from .feature_spec import FeatureSpec

# Default architecture used by the paper: a tiny MLP for tabular recourse.
_DEFAULT_POLICY_KWARGS: dict[str, Any] = {"net_arch": [8, 8]}


@dataclass
class Explanation:
    """Result of calling :meth:`FastAR.explain` on a single instance.

    Attributes
    ----------
    original:
        The input instance, in the same scaled space the agent was trained
        on. Shape ``(n_features,)``.
    counterfactual:
        The final state reached by the agent. Shape ``(n_features,)``.
    trajectory:
        The sequence of states visited, including ``original`` as the first
        row and ``counterfactual`` as the last. Shape
        ``(num_steps + 1, n_features)``.
    success:
        ``True`` iff the agent reached the target class before hitting the
        step budget.
    num_steps:
        Number of actions taken. Equal to ``trajectory.shape[0] - 1``.
    final_reward:
        Reward received on the final step.
    """

    original: np.ndarray
    counterfactual: np.ndarray
    trajectory: np.ndarray
    success: bool
    num_steps: int
    final_reward: float
    info: dict[str, Any] = field(default_factory=dict)


class FastAR:
    """High-level FastAR counterfactual explainer.

    Parameters
    ----------
    classifier:
        Any scikit-learn-like classifier with a ``predict_proba`` method.
        Must have been trained on data already scaled to ``[-1, 1]``.
    feature_spec:
        :class:`FeatureSpec` describing the feature layout and constraints.
    dist_lambda:
        Coefficient of the manifold-distance penalty in the reward.
    target_class:
        The class label FastAR is trying to reach. Defaults to ``1``.
    max_episode_steps:
        Maximum number of actions per training/evaluation episode.
    policy_kwargs:
        Forwarded to Stable-Baselines3's :class:`~stable_baselines3.PPO`.
        Defaults to a small two-layer MLP (``[8, 8]``).
    seed:
        Random seed forwarded to both the Gymnasium environment and PPO.

    Notes
    -----
    The classifier and the training data passed to :meth:`fit` must already
    live in the same ``[-1, 1]`` scaled space. FastAR does not perform any
    scaling internally — this keeps the package independent of any specific
    preprocessing pipeline.
    """

    def __init__(
        self,
        classifier: _ProbabilisticClassifier,
        feature_spec: FeatureSpec,
        *,
        dist_lambda: float = 0.1,
        target_class: int = 1,
        max_episode_steps: int = 50,
        policy_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self.classifier = classifier
        self.feature_spec = feature_spec
        self.dist_lambda = float(dist_lambda)
        self.target_class = int(target_class)
        self.max_episode_steps = int(max_episode_steps)
        self.policy_kwargs = (
            dict(_DEFAULT_POLICY_KWARGS)
            if policy_kwargs is None
            else dict(policy_kwargs)
        )
        self.seed = seed

        self._env: CounterfactualEnv | None = None
        self._model: PPO | None = None
        self._training_data: np.ndarray | None = None

    # ------------------------------------------------------------------ Fit

    def fit(
        self,
        X_train: np.ndarray,
        *,
        total_timesteps: int = 1_000_000,
        ppo_kwargs: dict[str, Any] | None = None,
        learn_kwargs: dict[str, Any] | None = None,
    ) -> "FastAR":
        """Train the FastAR policy.

        Parameters
        ----------
        X_train:
            Training data of shape ``(n_samples, n_features)``, already
            scaled to ``[-1, 1]``. Episodes start from random rows of this
            array during training.
        total_timesteps:
            Total number of environment steps to train for.
        ppo_kwargs:
            Extra keyword arguments forwarded to
            :class:`stable_baselines3.PPO`'s constructor.
        learn_kwargs:
            Extra keyword arguments forwarded to
            :meth:`stable_baselines3.PPO.learn`.
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        self._training_data = X_train
        self._env = self._make_env(X_train)

        ppo_kwargs = dict(ppo_kwargs or {})
        ppo_kwargs.setdefault("policy_kwargs", self.policy_kwargs)
        if self.seed is not None:
            ppo_kwargs.setdefault("seed", self.seed)

        self._model = PPO("MlpPolicy", self._env, **ppo_kwargs)
        self._model.learn(total_timesteps=total_timesteps, **(learn_kwargs or {}))
        return self

    # -------------------------------------------------------------- Explain

    def explain(
        self,
        x: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> Explanation:
        """Generate a counterfactual explanation for a single instance.

        Parameters
        ----------
        x:
            The instance to explain, shape ``(n_features,)``, in the same
            scaled space used for training.
        deterministic:
            If ``True`` (default), use the greedy action at each step.
            If ``False``, sample from the policy's action distribution.
        """
        if self._model is None or self._env is None:
            raise RuntimeError("FastAR.explain called before fit")

        x = np.asarray(x, dtype=np.float32).reshape(-1)
        env = self._env
        env.set_pinned_target(x)
        try:
            obs, _ = env.reset()
            trajectory: list[np.ndarray] = [obs.copy()]
            terminated = False
            truncated = False
            reward = 0.0
            while not (terminated or truncated):
                action, _ = self._model.predict(obs, deterministic=deterministic)
                obs, step_reward, terminated, truncated, _ = env.step(int(action))
                trajectory.append(obs.copy())
                reward = step_reward
        finally:
            env.set_pinned_target(None)

        trajectory_arr = np.stack(trajectory, axis=0)
        return Explanation(
            original=x.copy(),
            counterfactual=trajectory_arr[-1].copy(),
            trajectory=trajectory_arr,
            success=bool(terminated),
            num_steps=trajectory_arr.shape[0] - 1,
            final_reward=float(reward),
        )

    def explain_batch(
        self,
        X: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> list[Explanation]:
        """Generate counterfactual explanations for each row of ``X``."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"explain_batch expects a 2D array, got shape {X.shape}")
        return [self.explain(row, deterministic=deterministic) for row in X]

    # ---------------------------------------------------------- Persistence

    def save(self, path: str | Path) -> None:
        """Save the trained PPO policy to ``path``.

        Only the policy is persisted. To reload, call :meth:`load` with the
        same ``classifier`` and ``feature_spec`` used to create this
        instance — those are *not* serialized because they are the user's
        responsibility.
        """
        if self._model is None:
            raise RuntimeError("FastAR.save called before fit")
        self._model.save(str(path))

    def load_policy(self, path: str | Path, X_train: np.ndarray) -> "FastAR":
        """Load a PPO policy previously saved with :meth:`save`.

        ``X_train`` is required because the environment needs the training
        data pool to reconstruct the manifold-distance KNN index and to
        serve as the reset distribution.
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        self._training_data = X_train
        self._env = self._make_env(X_train)
        self._model = PPO.load(str(path), env=self._env)
        return self

    # --------------------------------------------------------------- Access

    @property
    def env(self) -> CounterfactualEnv:
        """The underlying :class:`CounterfactualEnv`.

        Available after :meth:`fit` or :meth:`load_policy`.
        """
        if self._env is None:
            raise RuntimeError("FastAR.env accessed before fit")
        return self._env

    @property
    def model(self) -> PPO:
        """The underlying Stable-Baselines3 PPO model.

        Available after :meth:`fit` or :meth:`load_policy`. Exposed for
        advanced users who want to poke at training internals.
        """
        if self._model is None:
            raise RuntimeError("FastAR.model accessed before fit")
        return self._model

    # -------------------------------------------------------------- Private

    def _make_env(self, X_train: np.ndarray) -> CounterfactualEnv:
        return CounterfactualEnv(
            classifier=self.classifier,
            training_data=X_train,
            feature_spec=self.feature_spec,
            dist_lambda=self.dist_lambda,
            target_class=self.target_class,
            max_episode_steps=self.max_episode_steps,
        )
