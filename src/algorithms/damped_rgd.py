"""Repeated Gradient Descent (RGD / Damped RGD) algorithm.

Executes gradient steps on the current empirical distribution:
theta_{t+1} = theta_t - eta * grad_theta R(theta_t; D_t)
"""

from typing import Dict, Optional, Union
import numpy as np
import torch

from src.algorithms.base import BaseRecommender


class DampedRGD(BaseRecommender):
    """Repeated Gradient Descent with step size eta and optional damping/decay."""

    def __init__(
        self,
        num_items: int = 50,
        dimension: int = 16,
        l2_reg: float = 1.0,
        lr: float = 0.05,
        num_steps: int = 1,
        damping: float = 1.0,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(num_items, dimension, l2_reg, device)
        self.lr = lr
        self.num_steps = num_steps
        self.damping = damping

    def fit(self, dataset: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Performs damped repeated gradient step on D_t."""
        theta_prev = self.theta.detach().clone()

        u = torch.from_numpy(dataset["user_features"]).float().to(self.device)
        items = torch.from_numpy(dataset["item_indices"]).long().to(self.device)
        y = torch.from_numpy(dataset["labels"]).float().to(self.device)

        if len(y) == 0:
            return {"loss": 0.0, "drift": 0.0, "status": "empty_data"}

        for _ in range(self.num_steps):
            if self.theta.grad is not None:
                self.theta.grad.zero_()
            loss = self.compute_loss(u, items, y)
            loss.backward()

            with torch.no_grad():
                grad = self.theta.grad
                # Damped update step
                self.theta.sub_(self.damping * self.lr * grad)

        drift = torch.norm(self.theta - theta_prev, p="fro").item()
        loss_val = float(loss.item())
        self.round_history.append({"loss": loss_val, "drift": float(drift)})

        return {
            "loss": loss_val,
            "drift": float(drift),
            "status": "step_taken",
        }
