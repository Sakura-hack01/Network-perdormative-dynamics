"""Causal Lookahead Performative Optimization algorithm.

Implements performative-optimal lookahead using cluster-randomized Two-Point Zero-Order
Gradient Estimation (Flaxman et al., Mendler-Dünner et al.) to minimize network interference
bias across graph communities, with a long-term welfare/diversity regularizer.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim

from src.algorithms.base import BaseRecommender


class CausalLookaheadPO(BaseRecommender):
    """Performative-Optimal Lookahead via Cluster-Randomized Zero-Order Estimation."""

    def __init__(
        self,
        num_items: int = 50,
        dimension: int = 16,
        l2_reg: float = 1.0,
        lr: float = 0.05,
        delta: float = 0.05,
        n_clusters: int = 8,
        div_weight: float = 0.05,
        seed: Optional[int] = 42,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(num_items, dimension, l2_reg, device)
        self.lr = lr
        self.delta = delta
        self.n_clusters = n_clusters
        self.div_weight = div_weight
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.cluster_assignments = None

    def partition_graph(self, adj_matrix: sp.csr_matrix) -> np.ndarray:
        """Partitions network adjacency matrix into balanced clusters using spectral partitioning.

        Minimizes cross-cluster edges (conductance / cut weight) to mitigate network
        interference spillover between treatment and control groups.
        """
        N = adj_matrix.shape[0]
        k = min(self.n_clusters, N)

        # Degree and Laplacian
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
        D_inv_sqrt = sp.diags(deg_inv_sqrt)

        # Normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}
        L_sym = sp.eye(N) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt

        # Smallest non-trivial eigenvectors of L_sym
        try:
            from scipy.sparse.linalg import eigsh
            vals, vecs = eigsh(L_sym, k=min(k + 1, N - 1), which="SM")
            features = vecs[:, 1:k]  # Skip trivial zero eigenvalue
            # Normalize rows
            row_norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / np.maximum(row_norms, 1e-12)

            from scipy.cluster.vq import kmeans2
            _, labels = kmeans2(features, k, minit="points", seed=self.seed)
        except Exception:
            # Fallback to balanced random/community assignment if graph is disconnected
            labels = np.arange(N) % k
            self.rng.shuffle(labels)

        self.cluster_assignments = labels
        return labels

    def compute_diversity_loss(
        self,
        user_features: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """Computes diversity regularizer to penalize item monopoly / echo collapse.

        Omega(theta) = sum_k (bar_p_k - 1/K)^2
        where bar_p_k is mean predicted activation of item k.
        """
        # Predicted scores: (N, K)
        scores = torch.matmul(user_features, theta.t())
        probs = torch.sigmoid(scores)  # (N, K)
        mean_item_exposure = torch.mean(probs, dim=0)  # (K,)
        target_uniform = 1.0 / self.num_items
        # Uniform dispersion penalty
        div_loss = torch.sum((mean_item_exposure - target_uniform) ** 2)
        return div_loss

    def lookahead_step(
        self,
        env,
        dataset: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Performs cluster-randomized two-point zero-order lookahead step.

        1. Partitions graph into clusters.
        2. Assigns half of clusters to +W, half to -W.
        3. Deploys perturbed policies, observing cluster-isolated empirical response.
        4. Calculates performative gradient g_ZO and diversity gradient.
        5. Updates theta.
        """
        theta_prev = self.theta.detach().clone()
        K, d = self.num_items, self.dimension
        dim_total = K * d

        # Partition graph if not already cached
        if self.cluster_assignments is None:
            self.partition_graph(env.population.adj_matrix)

        clusters = self.cluster_assignments
        n_c = self.n_clusters

        # Random perturbation direction W on unit Frobenius sphere
        Z = torch.randn(K, d, device=self.device)
        W = Z / torch.norm(Z, p="fro")

        # Cluster randomization: assign clusters randomly to +1 or -1
        cluster_signs = np.array([1 if c % 2 == 0 else -1 for c in range(n_c)])
        self.rng.shuffle(cluster_signs)
        user_signs = cluster_signs[clusters]  # (N,)

        u_raw = env.population.preferences.copy()
        N = len(u_raw)

        # Build policies for plus and minus clusters
        theta_plus = self.theta + self.delta * W
        theta_minus = self.theta - self.delta * W

        theta_plus_np = theta_plus.detach().cpu().numpy()
        theta_minus_np = theta_minus.detach().cpu().numpy()

        plus_mask = (user_signs == 1)
        minus_mask = (user_signs == -1)

        # Compute empirical losses on opposite clusters
        # Group Plus
        loss_plus = 0.0
        if np.any(plus_mask):
            slates_plus = env.compute_slates(theta_plus_np, u_raw[plus_mask])
            p_plus, b_plus = env.compute_choice_probabilities(slates_plus, u_raw[plus_mask])
            c_plus, _ = env.sample_choices(slates_plus, p_plus, b_plus)
            # Binary cross-entropy on plus deployment
            u_pt = torch.from_numpy(u_raw[plus_mask]).float().to(self.device)
            # Slates to items
            M = slates_plus.shape[1]
            u_rep = u_pt.repeat_interleave(M, dim=0)
            items_flat = torch.from_numpy(slates_plus.flatten()).long().to(self.device)
            lbls = np.zeros(len(u_raw[plus_mask]) * M, dtype=np.float32)
            for i_local, chosen in enumerate(c_plus):
                if chosen != -1:
                    pos = np.where(slates_plus[i_local] == chosen)[0]
                    if len(pos) > 0:
                        lbls[i_local * M + pos[0]] = 1.0
            lbls_t = torch.from_numpy(lbls).float().to(self.device)
            loss_plus = self.compute_loss(u_rep, items_flat, lbls_t, theta=theta_plus).item()

        # Group Minus
        loss_minus = 0.0
        if np.any(minus_mask):
            slates_minus = env.compute_slates(theta_minus_np, u_raw[minus_mask])
            p_minus, b_minus = env.compute_choice_probabilities(slates_minus, u_raw[minus_mask])
            c_minus, _ = env.sample_choices(slates_minus, p_minus, b_minus)
            u_mt = torch.from_numpy(u_raw[minus_mask]).float().to(self.device)
            M = slates_minus.shape[1]
            u_rep_m = u_mt.repeat_interleave(M, dim=0)
            items_flat_m = torch.from_numpy(slates_minus.flatten()).long().to(self.device)
            lbls_m = np.zeros(len(u_raw[minus_mask]) * M, dtype=np.float32)
            for i_local, chosen in enumerate(c_minus):
                if chosen != -1:
                    pos = np.where(slates_minus[i_local] == chosen)[0]
                    if len(pos) > 0:
                        lbls_m[i_local * M + pos[0]] = 1.0
            lbls_mt = torch.from_numpy(lbls_m).float().to(self.device)
            loss_minus = self.compute_loss(u_rep_m, items_flat_m, lbls_mt, theta=theta_minus).item()

        # Two-Point Zero-Order Directional Derivative:
        # g_ZO = (loss_plus - loss_minus) / (2 * delta) * dim * W
        diff_loss = loss_plus - loss_minus
        g_zo = (diff_loss / (2.0 * self.delta)) * float(dim_total) * W

        # Standard empirical risk gradient on dataset
        u_base = torch.from_numpy(dataset["user_features"]).float().to(self.device)
        items_base = torch.from_numpy(dataset["item_indices"]).long().to(self.device)
        y_base = torch.from_numpy(dataset["labels"]).float().to(self.device)

        if self.theta.grad is not None:
            self.theta.grad.zero_()
        base_loss = self.compute_loss(u_base, items_base, y_base)

        # Diversity loss gradient
        u_pop = torch.from_numpy(u_raw).float().to(self.device)
        div_loss = self.compute_diversity_loss(u_pop, self.theta)
        total_obj = base_loss + self.div_weight * div_loss
        total_obj.backward()

        # Perform lookahead update
        with torch.no_grad():
            standard_grad = self.theta.grad
            # Total performative gradient = standard empirical grad + zero-order performative shift
            total_grad = standard_grad + 0.1 * g_zo
            self.theta.sub_(self.lr * total_grad)

        drift = torch.norm(self.theta - theta_prev, p="fro").item()
        loss_val = float(base_loss.item())
        self.round_history.append({"loss": loss_val, "drift": float(drift)})

        return {
            "loss": loss_val,
            "drift": float(drift),
            "diversity_loss": float(div_loss.item()),
            "status": "lookahead_updated",
        }

    def fit(self, dataset: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Fallback fit for standard API compliance without env reference."""
        u_base = torch.from_numpy(dataset["user_features"]).float().to(self.device)
        items_base = torch.from_numpy(dataset["item_indices"]).long().to(self.device)
        y_base = torch.from_numpy(dataset["labels"]).float().to(self.device)

        theta_prev = self.theta.detach().clone()
        if self.theta.grad is not None:
            self.theta.grad.zero_()
        loss = self.compute_loss(u_base, items_base, y_base)
        loss.backward()

        with torch.no_grad():
            self.theta.sub_(self.lr * self.theta.grad)

        drift = torch.norm(self.theta - theta_prev, p="fro").item()
        return {"loss": float(loss.item()), "drift": float(drift), "status": "standard_step"}
