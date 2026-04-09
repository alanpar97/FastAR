# FastAR

Amortized generation of sequential algorithmic recourses for black-box models.

FastAR trains a PPO agent to edit an input instance — one feature at a time — until a black-box classifier flips its prediction to a desired class. The agent learns a general recourse policy from training data, so at inference time it can produce counterfactual explanations for new instances without re-optimization.

This package is a clean refactor of the [original research code](https://github.com/vsahil/FastAR-RL-for-generating-AR) into a reusable Python library. The experiment harness, baselines, and dataset-specific scaffolding have been removed; what remains is the core method behind a simple `fit` / `explain` API.

## Installation

```bash
git clone https://github.com/alanpar97/FastAR.git
cd FastAR

uv venv
uv pip install -e .

# If you want to run the examples (requires pandas):
uv pip install -e ".[examples]"
```

## Quick start

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from fastar import FastAR, FeatureSpec

# 1. Train any classifier with predict_proba
X = np.random.default_rng(0).normal(size=(200, 3)).astype(np.float32)
X = np.clip(X / X.max(axis=0), -1, 1)  # scale to [-1, 1]
y = (X[:, 0] + X[:, 1] > 0).astype(int)
classifier = LogisticRegression().fit(X, y)

# 2. Describe your features
spec = FeatureSpec(
    columns=("income", "age", "score"),
    continuous=("income", "age", "score"),
    immutable=("age",),               # age cannot be changed
    monotonic_increasing=("score",),   # score can only go up
)

# 3. Fit and explain
fastar = FastAR(classifier, spec, dist_lambda=0.1, seed=42)
fastar.fit(X, total_timesteps=50_000)

x = X[classifier.predict(X) == 0][0]  # pick an undesirable instance
result = fastar.explain(x)

print(f"success: {result.success}")
print(f"steps:   {result.num_steps}")
print(f"before:  {result.original}")
print(f"after:   {result.counterfactual}")
```

## Usage guide

### Bring your own model and data

FastAR works with **any classifier** that exposes a scikit-learn-style `predict_proba(X)` method. It does not train a classifier for you — you bring one that is already fitted.

**Data must be scaled to `[-1, 1]`** before passing it to FastAR. The library does not scale internally so that it stays independent of your preprocessing pipeline. A typical setup:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
X_train_scaled = scaler.transform(X_train).astype(np.float32)

# Train your classifier on the scaled data
classifier.fit(X_train_scaled, y_train)
```

### Defining a FeatureSpec

`FeatureSpec` encodes domain knowledge about your dataset. This is how you tell FastAR which features can change, which cannot, and how features relate to each other.

```python
from fastar import FeatureSpec

spec = FeatureSpec(
    columns=("age", "education", "income", "sex", "hours_per_week"),
    continuous=("age", "income", "hours_per_week"),  # numerical features
    immutable=("sex",),                               # cannot be changed
    monotonic_increasing=("age", "education"),         # can only increase
    correlated=(
        # increasing education causes age to increase by 0.054 in scaled space
        ("education", "age", 0.054),
    ),
    step_size=0.05,  # magnitude of each action in [-1, 1] space
)
```

Columns not listed in `continuous` are treated as categorical. The agent still edits them by the same step size — the distinction is for your own bookkeeping.

### Training

```python
fastar = FastAR(
    classifier=classifier,
    feature_spec=spec,
    dist_lambda=0.1,        # manifold-distance penalty coefficient
    max_episode_steps=50,   # max actions per episode
    seed=0,
)

fastar.fit(
    X_train_scaled,
    total_timesteps=500_000,  # more steps = better policy
)
```

Under the hood, `fit` creates a Gymnasium environment and trains a [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) PPO agent. You can pass extra arguments to the PPO constructor or the `.learn()` call:

```python
fastar.fit(
    X_train_scaled,
    total_timesteps=1_000_000,
    ppo_kwargs={"verbose": 1, "n_steps": 256},
    learn_kwargs={"log_interval": 50},
)
```

### Explaining instances

```python
# Single instance
explanation = fastar.explain(x_scaled)

print(explanation.success)          # did the classifier flip?
print(explanation.num_steps)        # how many actions were taken
print(explanation.original)         # starting point
print(explanation.counterfactual)   # final point
print(explanation.trajectory)       # (num_steps+1, n_features) array
```

```python
# Batch
explanations = fastar.explain_batch(X_test_scaled)
validity = sum(e.success for e in explanations) / len(explanations)
```

### Saving and loading policies

```python
# Save after training
fastar.save("my_policy")

# Load into a new FastAR instance (same classifier and spec required)
fastar2 = FastAR(classifier, spec, dist_lambda=0.1)
fastar2.load_policy("my_policy.zip", X_train_scaled)
explanation = fastar2.explain(x)
```

## Examples

Full runnable examples are in the [`examples/`](examples/) directory:

- **[`quickstart.py`](examples/quickstart.py)** — Minimal 2D synthetic dataset. Runs in seconds.
- **[`german_credit.py`](examples/german_credit.py)** — End-to-end pipeline on the German Credit dataset: loading data, training an MLP, defining feature constraints, fitting FastAR, and printing the counterfactual.

```bash
python examples/quickstart.py
python examples/german_credit.py
```

## Project structure

```
FastAR/
├── src/fastar/
│   ├── __init__.py        # public exports: FastAR, FeatureSpec, CounterfactualEnv, Explanation
│   ├── agent.py           # FastAR class (fit / explain / save / load)
│   ├── env.py             # CounterfactualEnv (Gymnasium environment)
│   └── feature_spec.py    # FeatureSpec dataclass
├── tests/
├── examples/
└── pyproject.toml
```

## Citation

If you use FastAR in your work, please cite the original paper:

```bibtex
@inproceedings{verma2022amortized,
  title={Amortized generation of sequential algorithmic recourses for black-box models},
  author={Verma, Sahil and Hines, Keegan and Dickerson, John P},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={8},
  pages={8512--8519},
  year={2022}
}
```
