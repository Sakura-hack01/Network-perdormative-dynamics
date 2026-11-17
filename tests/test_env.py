"""Unit tests for NetworkPopulation, StatefulFeedbackEnv, and CausalLookaheadPO."""

import sys
from pathlib import Path
import numpy as np
import pytest
import scipy.sparse as sp
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.env.network_population import NetworkPopulation
from src.env.stateful_feedback import StatefulFeedbackEnv, ItemCatalog
from src.algorithms.causal_lookahead_po import CausalLookaheadPO


def test_network_population_properties():
    """Tests SBM generation, degree normalization, and preference bounds."""
    N = 200
    d = 4
    pop = NetworkPopulation(
        num_users=N,
        dimension=d,
        num_communities=2,
        community_sep=0.6,
        graph_type="sbm",
        p_in=0.1,
        p_out=0.01,
        seed=123,
    )

    # 1. Balanced community assignments
    c0 = np.sum(pop.community_assignments == 0)
    c1 = np.sum(pop.community_assignments == 1)
    assert c0 == N // 2
    assert c1 == N // 2

    # 2. Row-stochastic normalization: row sums must equal 1.0
    row_sums = np.array(pop.norm_adj_matrix.sum(axis=1)).flatten()
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

    # 3. Preference bounds: all anchors must satisfy ||u||_2 <= 1.0 + eps
    norms = np.linalg.norm(pop.anchors, axis=1)
    assert np.all(norms <= 1.0001)

    # 4. Community separation > 0
    sep = pop.get_community_separation()
    assert sep > 0.4


def test_multinomial_logit_conservation():
    """Verifies that click probabilities plus bounce probability strictly sum to 1.0."""
    N = 50
    d = 8
    K = 25
    M = 5

    pop = NetworkPopulation(num_users=N, dimension=d, seed=42)
    cat = ItemCatalog(num_items=K, dimension=d, seed=42)
    env = StatefulFeedbackEnv(population=pop, catalog=cat, slate_size=M, beta=2.0, tau_0=0.1)

    theta = np.random.randn(K, d) * 0.2
    slates = env.compute_slates(theta)
    click_p, bounce_p = env.compute_choice_probabilities(slates)

    total_prob = np.sum(click_p, axis=1) + bounce_p
    np.testing.assert_allclose(total_prob, 1.0, rtol=1e-5)


def test_stateful_memory_and_peer_influence():
    """Verifies state update equation with lambda, kappa, and performative shift."""
    N = 100
    d = 4
    K = 10
    M = 2

    pop = NetworkPopulation(num_users=N, dimension=d, seed=99)
    cat = ItemCatalog(num_items=K, dimension=d, seed=99)

    lambda_mem = 0.4
    kappa_peer = 0.3
    eps_perf = 0.0

    env = StatefulFeedbackEnv(
        population=pop,
        catalog=cat,
        slate_size=M,
        lambda_mem=lambda_mem,
        kappa_peer=kappa_peer,
        epsilon_perf=eps_perf,
        transition_noise_std=0.0,
        seed=99,
    )

    u0 = pop.anchors.copy()
    theta = np.zeros((K, d))
    info = env.step(theta, deterministic=True)
    u1 = env.population.preferences

    # When eps=0, noise=0, u_0 = u_t, peer influence A_tilde u_0
    peer = pop.norm_adj_matrix.dot(u0)
    expected_u1 = (1.0 - lambda_mem - kappa_peer) * u0 + lambda_mem * u0 + kappa_peer * peer
    expected_u1 = pop.project_preferences(expected_u1)

    np.testing.assert_allclose(u1, expected_u1, atol=1e-5)


def test_causal_lookahead_spectral_clustering():
    """Verifies that spectral partitioning in CausalLookaheadPO generates balanced clusters."""
    N = 100
    d = 4
    K = 10
    pop = NetworkPopulation(num_users=N, dimension=d, graph_type="sbm", seed=7)
    cat = ItemCatalog(num_items=K, dimension=d, seed=7)
    env = StatefulFeedbackEnv(population=pop, catalog=cat, slate_size=3, seed=7)

    lookahead = CausalLookaheadPO(num_items=K, dimension=d, n_clusters=4, seed=7)
    clusters = lookahead.partition_graph(pop.adj_matrix)

    assert len(clusters) == N
    assert len(np.unique(clusters)) == 4
    assert np.all(np.bincount(clusters) > 0)
