"""Evaluation metrics for performative prediction and dynamical systems."""

from typing import Dict, List, Optional, Union
import numpy as np
import torch
import scipy.stats as stats

from src.algorithms.base import BaseRecommender
from src.env.network_population import NetworkPopulation
from src.env.stateful_feedback import StatefulFeedbackEnv


def compute_empirical_risk(
    model: BaseRecommender,
    dataset: Dict[str, np.ndarray],
) -> float:
    """Computes regularized empirical risk R_hat(theta; D_t) on batch D_t."""
    with torch.no_grad():
        loss = model.compute_loss(
            dataset["user_features"],
            dataset["item_indices"],
            dataset["labels"],
        )
        return float(loss.item())


def compute_performative_risk(
    model: BaseRecommender,
    env: StatefulFeedbackEnv,
    num_eval_samples: int = 500,
) -> float:
    """Computes true post-deployment performative risk PR(theta) under induced state.

    Evaluates regularized cross-entropy loss on fresh feedback from the deployed model.
    """
    theta_np = model.get_theta()
    u_curr = env.population.preferences

    # Subsample if population is very large, or evaluate on full population
    if len(u_curr) > num_eval_samples:
        idx = np.random.choice(len(u_curr), size=num_eval_samples, replace=False)
        u_eval = u_curr[idx]
    else:
        u_eval = u_curr

    slates = env.compute_slates(theta_np, u_eval)
    p_click, p_bounce = env.compute_choice_probabilities(slates, u_eval)
    c_items, _ = env.sample_choices(slates, p_click, p_bounce)

    N_eval, M = slates.shape
    u_pt = torch.from_numpy(u_eval).float().to(model.device)
    u_rep = u_pt.repeat_interleave(M, dim=0)
    items_flat = torch.from_numpy(slates.flatten()).long().to(model.device)

    labels = np.zeros(N_eval * M, dtype=np.float32)
    for i in range(N_eval):
        c = c_items[i]
        if c != -1:
            pos = np.where(slates[i] == c)[0]
            if len(pos) > 0:
                labels[i * M + pos[0]] = 1.0

    labels_pt = torch.from_numpy(labels).float().to(model.device)

    with torch.no_grad():
        loss = model.compute_loss(u_rep, items_flat, labels_pt)
        return float(loss.item())


def compute_parameter_drift(
    theta_curr: np.ndarray,
    theta_prev: np.ndarray,
) -> float:
    """Computes parameter drift ||theta_{t+1} - theta_t||_F."""
    return float(np.linalg.norm(theta_curr - theta_prev, ord="fro"))


def compute_polarization_metrics(
    population: NetworkPopulation,
) -> Dict[str, float]:
    """Computes polarization and echo-chamber metrics:

    - Sarle's Bimodality Coefficient along primary axis
    - Community mean Euclidean separation
    - Total preference variance/dispersion
    """
    u = population.preferences
    bimodality = population.get_bimodality_coefficient(axis=0, u=u)
    community_sep = population.get_community_separation(u=u)
    pref_var = population.get_preference_variance(u=u)

    return {
        "bimodality_coefficient": float(bimodality),
        "community_separation": float(community_sep),
        "preference_variance": float(pref_var),
    }


def compute_welfare_and_diversity(
    env: StatefulFeedbackEnv,
    slates: np.ndarray,
    clicked_items: np.ndarray,
) -> Dict[str, float]:
    """Computes social welfare, bounce rate, and recommendation catalog diversity.

    - Welfare: Average utility realized by users
    - Bounce rate: Fraction of outside-option choices
    - Catalog entropy: Shannon entropy of slate item frequencies
    """
    N = len(clicked_items)
    u_curr = env.population.preferences

    # Utility realized
    utilities = np.zeros(N)
    for i in range(N):
        c = clicked_items[i]
        if c != -1:
            utilities[i] = float(np.dot(u_curr[i], env.catalog.items_np[c]))
        else:
            utilities[i] = env.tau_0

    welfare = float(np.mean(utilities))
    bounce_rate = float(np.mean(clicked_items == -1))

    # Catalog exposure entropy
    all_shown = slates.flatten()
    counts = np.bincount(all_shown, minlength=env.catalog.num_items)
    probs = counts / np.maximum(counts.sum(), 1.0)
    nonzero_p = probs[probs > 0]
    catalog_entropy = float(-np.sum(nonzero_p * np.log(nonzero_p)))

    return {
        "social_welfare": welfare,
        "bounce_rate": bounce_rate,
        "catalog_entropy": catalog_entropy,
    }


def evaluate_round_metrics(
    round_idx: int,
    model: BaseRecommender,
    env: StatefulFeedbackEnv,
    dataset: Dict[str, np.ndarray],
    step_info: Dict,
    theta_prev: Optional[np.ndarray] = None,
) -> Dict[str, Union[int, float]]:
    """Aggregates all system metrics for logging at round t."""
    theta_curr = model.get_theta()
    emp_risk = compute_empirical_risk(model, dataset)
    perf_risk = compute_performative_risk(model, env)

    if theta_prev is not None:
        drift = compute_parameter_drift(theta_curr, theta_prev)
    else:
        drift = 0.0

    pol_metrics = compute_polarization_metrics(env.population)
    welf_metrics = compute_welfare_and_diversity(
        env,
        step_info["slates"],
        step_info["clicked_items"],
    )

    return {
        "round": round_idx,
        "empirical_risk": emp_risk,
        "performative_risk": perf_risk,
        "parameter_drift": drift,
        **pol_metrics,
        **welf_metrics,
    }
