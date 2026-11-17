"""Static Empirical Risk Minimization (ERM) algorithm.

Fits model parameters theta_0 on initial distribution D_0 and freezes them,
serving as the non-adaptive baseline in performative prediction.
"""

from typing import Dict, Optional, Union
import numpy as np
import torch
import torch.optim as optim

from src.algorithms.base import BaseRecommender


class StaticERM(BaseRecommender):
    """Static ERM: Fits theta once at round 0 and remains fixed thereafter."""

    def __init__(
        self,
        num_items: int = 50,
        dimension: int = 16,
        l2_reg: float = 1.0,
        max_iter: int = 50,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(num_items, dimension, l2_reg, device)
        self.max_iter = max_iter
        self.is_fitted = False

    def fit(self, dataset: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Fits theta only if not already fitted. Subsequent calls are no-ops."""
        if self.is_fitted:
            loss_val = float(self.compute_loss(
                dataset["user_features"],
                dataset["item_indices"],
                dataset["labels"],
            ).item())
            return {"loss": loss_val, "status": "frozen"}

        u = torch.from_numpy(dataset["user_features"]).float().to(self.device)
        items = torch.from_numpy(dataset["item_indices"]).long().to(self.device)
        y = torch.from_numpy(dataset["labels"]).float().to(self.device)

        optimizer = optim.LBFGS(
            [self.theta],
            lr=1.0,
            max_iter=self.max_iter,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = self.compute_loss(u, items, y)
            loss.backward()
            return loss

        final_loss = optimizer.step(closure)
        self.is_fitted = True

        return {"loss": float(final_loss.item()), "status": "converged"}
