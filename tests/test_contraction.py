"""Unit tests for theoretical contraction guarantees and breakdown in RRM.

Verifies:
1. When lambda=0, kappa=0, and epsilon < gamma / beta, Repeated Risk Minimization (RRM)
   converges linearly in ||theta_{t+1} - theta_t||_F matching Perdomo et al. (ICML 2020)
   contraction bounds.
2. When epsilon > gamma / beta, contraction breaks down.
3. Loss function strong convexity and gradient consistency.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.env.network_population import NetworkPopulation
from src.env.stateful_feedback import StatefulFeedbackEnv, ItemCatalog
from src.algorithms.rrm import RepeatedRiskMinimization


def test_contraction_guarantee_linear_convergence():
    """Verifies geometric / linear contraction when lambda=0, kappa=0, epsilon < gamma/beta."""
    seed = 42
    N = 400
    d = 4
    K = 10
    M = 4

    l2_reg = 2.0     # gamma = 2.0
    beta = 1.0       # beta = 1.0
    # Theoretical threshold gamma / beta = 2.0
    epsilon = 0.05   # Well inside contraction region: epsilon << gamma / beta

    pop = NetworkPopulation(
        num_users=N,
        dimension=d,
        num_communities=2,
        community_sep=0.5,
        anchor_noise_std=0.1,
        graph_type="sbm",
        p_in=0.08,
        p_out=0.005,
        seed=seed,
    )
    catalog = ItemCatalog(num_items=K, dimension=d, seed=seed, community_sep=0.5)

    env = StatefulFeedbackEnv(
        population=pop,
        catalog=catalog,
        slate_size=M,
        num_neg_samples=K - M,  # Full candidate evaluation ensures smooth convex mapping
        beta=beta,
        tau_0=0.0,
        lambda_mem=0.0,  # Memoryless
        kappa_peer=0.0,  # No peer interference
        epsilon_perf=epsilon,
        transition_noise_std=0.0,  # Deterministic expectation for contractive mapping
        seed=seed,
    )

    rrm = RepeatedRiskMinimization(num_items=K, dimension=d, l2_reg=l2_reg, max_iter=40)
    rng = np.random.RandomState(seed)
    theta_0 = rng.randn(K, d) * 0.2
    rrm.init_theta(theta_0)

    drifts = []
    rounds = 8

    for t in range(rounds):
        theta_curr = rrm.get_theta().copy()
        step_res = env.step(theta_curr, deterministic=True)
        rrm.fit(step_res["dataset"])
        theta_next = rrm.get_theta()
        drift = float(np.linalg.norm(theta_next - theta_curr))
        drifts.append(drift)

    drifts = np.array(drifts)

    # 1. Parameter drift should decay geometrically towards 0
    assert len(drifts) == rounds, f"Expected {rounds} drift observations."
    assert drifts[-1] < drifts[0] * 0.05, (
        f"Drift must decrease significantly under contraction. Initial drift: {drifts[0]:.6f}, Final drift: {drifts[-1]:.6f}"
    )
    assert drifts[-1] < 0.005, f"Final drift should reach convergence near 0. Got: {drifts[-1]:.6f}"

    # 2. Sequential contraction: drift at round 1 must contract sharply from initial deployment
    assert drifts[1] < drifts[0] * 0.1, (
        f"Contraction requires immediate geometric decay. Round 0: {drifts[0]:.4f}, Round 1: {drifts[1]:.4f}"
    )


def test_contraction_breakdown_high_sensitivity():
    """Verifies that large epsilon > gamma / beta causes breakdown of contraction."""
    seed = 42
    N = 400
    d = 4
    K = 10
    M = 4

    l2_reg = 0.5   # gamma = 0.5
    beta = 2.0     # beta = 2.0 -> threshold gamma / beta = 0.25
    epsilon = 0.8  # Well above contraction threshold

    pop = NetworkPopulation(
        num_users=N,
        dimension=d,
        num_communities=2,
        community_sep=0.5,
        anchor_noise_std=0.1,
        graph_type="sbm",
        p_in=0.08,
        p_out=0.005,
        seed=seed,
    )
    catalog = ItemCatalog(num_items=K, dimension=d, seed=seed, community_sep=0.5)

    env = StatefulFeedbackEnv(
        population=pop,
        catalog=catalog,
        slate_size=M,
        num_neg_samples=K - M,
        beta=beta,
        tau_0=0.0,
        lambda_mem=0.0,
        kappa_peer=0.0,
        epsilon_perf=epsilon,
        transition_noise_std=0.0,
        seed=seed,
    )

    rrm = RepeatedRiskMinimization(num_items=K, dimension=d, l2_reg=l2_reg, max_iter=30)
    theta_0 = np.random.RandomState(seed).randn(K, d) * 0.3
    rrm.init_theta(theta_0)

    drifts = []
    rounds = 8

    for t in range(rounds):
        theta_curr = rrm.get_theta().copy()
        step_res = env.step(theta_curr, deterministic=True)
        rrm.fit(step_res["dataset"])
        theta_next = rrm.get_theta()
        drift = float(np.linalg.norm(theta_next - theta_curr))
        drifts.append(drift)

    drifts = np.array(drifts)
    # Under large epsilon, final drift does NOT decay to 0
    assert drifts[-1] > 0.01, f"Drift should remain substantial under breakdown. Got: {drifts[-1]}"


def test_regularized_loss_strong_convexity_and_optimality():
    """Tests that RRM solves ERM to an exact stationary point with zero gradient norm."""
    N = 100
    d = 6
    K = 10
    M = 3

    u = np.random.randn(N * M, d).astype(np.float32)
    items = np.random.randint(0, K, size=N * M)
    y = np.random.randint(0, 2, size=N * M).astype(np.float32)

    dataset = {"user_features": u, "item_indices": items, "labels": y}

    rrm = RepeatedRiskMinimization(num_items=K, dimension=d, l2_reg=1.0, max_iter=50)
    rrm.fit(dataset)

    # Check gradient at convergence
    rrm.theta.requires_grad_(True)
    if rrm.theta.grad is not None:
        rrm.theta.grad.zero_()
    loss = rrm.compute_loss(u, items, y)
    loss.backward()

    grad_norm = torch.norm(rrm.theta.grad, p="fro").item()
    assert grad_norm < 1e-4, f"L-BFGS should converge to stationary point. Grad norm: {grad_norm}"
