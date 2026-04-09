"""FastAR — reinforcement-learning-based counterfactual recourse.

Public API::

    from fastar import FastAR, CounterfactualEnv, FeatureSpec, Explanation
"""

from .agent import Explanation, FastAR
from .env import CounterfactualEnv
from .feature_spec import FeatureSpec

__all__ = [
    "CounterfactualEnv",
    "Explanation",
    "FastAR",
    "FeatureSpec",
]
