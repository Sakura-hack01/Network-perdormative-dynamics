"""Recommender algorithms package for performative prediction dynamics."""

from src.algorithms.base import BaseRecommender
from src.algorithms.static_erm import StaticERM
from src.algorithms.rrm import RepeatedRiskMinimization
from src.algorithms.damped_rgd import DampedRGD
from src.algorithms.causal_lookahead_po import CausalLookaheadPO

__all__ = [
    "BaseRecommender",
    "StaticERM",
    "RepeatedRiskMinimization",
    "DampedRGD",
    "CausalLookaheadPO",
]
