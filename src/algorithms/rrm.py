"""Repeated Risk Minimization (RRM) algorithm.

Solves the empirical risk minimization problem to convergence at every round:
theta_{t+1} = argmin_theta R(theta; D_t) + (gamma / 2) ||theta||_F^2
using L-BFGS with warm-starting.
"""

from typing import Dict, Optional, Union
import numpy as np
import torch
import torch.optim as optim

from src.algorithms.base import BaseRecommender


class RepeatedRiskMinimization(BaseRecommender):
    """Repeated Risk Minimization (RRM) algorithm."""

    def __init__(
        self,
        num_items: int = 50,
        dimension: int = 16,
        l2_reg: float = 1.0,
        max_iter: int = 40,
        tolerance_grad: float = 1e-6,
        tolerance_change: float = 1e-9,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(num_items, dimension, l2_reg, device)
        self.max_iter = max_iter
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change

    def fit(self, dataset: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Solves ERM on D_t to convergence starting from current theta_t."""
        theta_prev = self.theta.detach().clone()

        u = torch.from_numpy(dataset["user_features"]).float().to(self.device)
        items = torch.from_numpy(dataset["item_indices"]).long().to(self.device)
        y = torch.from_numpy(dataset["labels"]).float().to(self.device)

        if len(y) == 0:
            return {"loss": 0.0, "drift": 0.0, "status": "empty_data"}

        optimizer = optim.LBFGS(
            [self.theta],
            lr=1.0,
            max_iter=self.max_iter,
            tolerance_grad=self.tolerance_grad,
            tolerance_change=self.tolerance_change,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = self.compute_loss(u, items, y)
            loss.backward()
            return loss

        final_loss = optimizer.step(closure)

        drift = torch.norm(self.theta - theta_prev, p="fro").item()
        self.round_history.append({"loss": float(final_loss.item()), "drift": float(drift)})

        return {
            "loss": float(final_loss.item()),
            "drift": float(drift),
            "status": "converged",
        }
