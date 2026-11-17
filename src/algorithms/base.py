"""Base recommender algorithm and optimization utilities."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class BaseRecommender(ABC, nn.Module):
    """Abstract base class for performative recommendation algorithms."""

    def __init__(
        self,
        num_items: int = 50,
        dimension: int = 16,
        l2_reg: float = 1.0,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.num_items = num_items
        self.dimension = dimension
        self.l2_reg = l2_reg
        self.device = torch.device(device)

        # Parameter matrix theta of shape (num_items, dimension)
        self.theta = nn.Parameter(torch.zeros(num_items, dimension, device=self.device))
        self.round_history = []

    def init_theta(self, theta_init: Optional[Union[np.ndarray, torch.Tensor]] = None) -> None:
        """Initializes parameter matrix theta."""
        if theta_init is not None:
            if isinstance(theta_init, np.ndarray):
                tensor_init = torch.from_numpy(theta_init).float().to(self.device)
            else:
                tensor_init = theta_init.float().to(self.device)
            with torch.no_grad():
                self.theta.copy_(tensor_init)
        else:
            with torch.no_grad():
                # Normalized Xavier/He-like small normal initialization
                self.theta.normal_(0.0, 1.0 / np.sqrt(self.dimension))

    def get_theta(self) -> np.ndarray:
        """Returns parameter matrix as a NumPy array."""
        return self.theta.detach().cpu().numpy().copy()

    def set_theta(self, theta_new: Union[np.ndarray, torch.Tensor]) -> None:
        """Sets parameter matrix from NumPy array or tensor."""
        if isinstance(theta_new, np.ndarray):
            tensor_new = torch.from_numpy(theta_new).float().to(self.device)
        else:
            tensor_new = theta_new.float().to(self.device)
        with torch.no_grad():
            self.theta.copy_(tensor_new)

    def predict_scores(self, user_features: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Computes predicted scores s_ik = <theta_k, u_i>."""
        if isinstance(user_features, np.ndarray):
            u = torch.from_numpy(user_features).float().to(self.device)
        else:
            u = user_features.float().to(self.device)
        return torch.matmul(u, self.theta.t())

    def compute_loss(
        self,
        user_features: Union[np.ndarray, torch.Tensor],
        item_indices: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        theta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes regularized binary cross-entropy loss:

        R(theta; D) = (1/|D|) sum_j BCE( <theta_{k_j}, x_j>, y_j ) + (gamma / 2) ||theta||_F^2
        """
        if theta is None:
            theta = self.theta

        if isinstance(user_features, np.ndarray):
            user_features = torch.from_numpy(user_features).float().to(self.device)
        if isinstance(item_indices, np.ndarray):
            item_indices = torch.from_numpy(item_indices).long().to(self.device)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).float().to(self.device)

        if len(labels) == 0:
            return 0.5 * self.l2_reg * torch.sum(theta ** 2)

        # Selected theta vectors for each example: (batch_size, d)
        selected_theta = theta[item_indices]
        logits = torch.sum(selected_theta * user_features, dim=-1)

        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
        reg_loss = 0.5 * self.l2_reg * torch.sum(theta ** 2)
        return bce_loss + reg_loss

    @abstractmethod
    def fit(self, dataset: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Updates model parameters given observed feedback dataset."""
        pass
