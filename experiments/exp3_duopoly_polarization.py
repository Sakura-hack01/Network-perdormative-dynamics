"""Experiment 3: Duopoly Competition and Accelerated Polarization.

Investigates how competition between two platforms (Platform A and Platform B)
for user attention accelerates community polarization, niche differentiation,
and parameter divergence ||theta_A - theta_B||_F under network interference.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.env.network_population import NetworkPopulation
from src.env.stateful_feedback import StatefulFeedbackEnv, ItemCatalog
from src.algorithms.rrm import RepeatedRiskMinimization
from src.metrics.evaluators import (
    compute_polarization_metrics,
    compute_parameter_drift,
    compute_empirical_risk,
)


def run_duopoly_experiment(
    output_dir: Path,
    num_users: int = 1000,
    dimension: int = 16,
    num_items: int = 50,
    slate_size: int = 5,
    beta: float = 2.5,
    selection_temp: float = 2.0,
    l2_reg: float = 1.0,
    rounds: int = 35,
    seed: int = 123,
) -> pd.DataFrame:
    """Simulates duopoly competition and compares with monopoly benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []

    print("=== Running Exp 3: Duopoly Competition vs Monopoly Benchmark ===")

    # 1. Duopoly Simulation
    print("  --> Running Duopoly Competition (Platform A vs Platform B)...")
    pop_duo = NetworkPopulation(
        num_users=num_users,
        dimension=dimension,
        num_communities=2,
        community_sep=0.6,
        anchor_noise_std=0.15,
        graph_type="sbm",
        p_in=0.08,
        p_out=0.005,
        seed=seed,
    )
    catalog = ItemCatalog(
        num_items=num_items,
        dimension=dimension,
        seed=seed,
        community_sep=0.6,
    )
    env_duo = StatefulFeedbackEnv(
        population=pop_duo,
        catalog=catalog,
        slate_size=slate_size,
        beta=beta,
        tau_0=0.0,
        lambda_mem=0.30,
        kappa_peer=0.25,
        epsilon_perf=0.25,
        transition_noise_std=0.015,
        seed=seed,
    )

    model_A = RepeatedRiskMinimization(num_items, dimension, l2_reg=l2_reg, max_iter=25)
    model_B = RepeatedRiskMinimization(num_items, dimension, l2_reg=l2_reg, max_iter=25)

    # Initial slight differentiation
    init_A = np.random.RandomState(seed).randn(num_items, dimension) * 0.1
    init_B = np.random.RandomState(seed + 1).randn(num_items, dimension) * 0.1
    model_A.init_theta(init_A)
    model_B.init_theta(init_B)

    theta_prev_A = None
    theta_prev_B = None

    for t in range(rounds):
        theta_A_curr = model_A.get_theta()
        theta_B_curr = model_B.get_theta()

        step_res = env_duo.step_duopoly(
            theta_A=theta_A_curr,
            theta_B=theta_B_curr,
            selection_temp=selection_temp,
        )

        pol = compute_polarization_metrics(env_duo.population)
        inter_platform_dist = float(np.linalg.norm(theta_A_curr - theta_B_curr, ord="fro"))

        drift_A = compute_parameter_drift(theta_A_curr, theta_prev_A) if theta_prev_A is not None else 0.0
        drift_B = compute_parameter_drift(theta_B_curr, theta_prev_B) if theta_prev_B is not None else 0.0

        risk_A = compute_empirical_risk(model_A, step_res["dataset_A"]) if len(step_res["dataset_A"]["labels"]) > 0 else 0.0
        risk_B = compute_empirical_risk(model_B, step_res["dataset_B"]) if len(step_res["dataset_B"]["labels"]) > 0 else 0.0

        record = {
            "round": t,
            "setting": "duopoly",
            "market_share_A": step_res["market_share_A"],
            "market_share_B": step_res["market_share_B"],
            "inter_platform_drift": inter_platform_dist,
            "drift_A": drift_A,
            "drift_B": drift_B,
            "empirical_risk_A": risk_A,
            "empirical_risk_B": risk_B,
            **pol,
        }
        records.append(record)

        theta_prev_A = theta_A_curr
        theta_prev_B = theta_B_curr

        if len(step_res["dataset_A"]["labels"]) > 0:
            model_A.fit(step_res["dataset_A"])
        if len(step_res["dataset_B"]["labels"]) > 0:
            model_B.fit(step_res["dataset_B"])

    # 2. Single Platform (Monopoly) Benchmark
    print("  --> Running Single Platform Monopoly Benchmark...")
    pop_mono = NetworkPopulation(
        num_users=num_users,
        dimension=dimension,
        num_communities=2,
        community_sep=0.6,
        anchor_noise_std=0.15,
        graph_type="sbm",
        p_in=0.08,
        p_out=0.005,
        seed=seed,
    )
    env_mono = StatefulFeedbackEnv(
        population=pop_mono,
        catalog=catalog,
        slate_size=slate_size,
        beta=beta,
        tau_0=0.0,
        lambda_mem=0.30,
        kappa_peer=0.25,
        epsilon_perf=0.25,
        transition_noise_std=0.015,
        seed=seed,
    )
    model_mono = RepeatedRiskMinimization(num_items, dimension, l2_reg=l2_reg, max_iter=25)
    model_mono.init_theta(init_A)

    theta_prev_mono = None
    for t in range(rounds):
        theta_curr = model_mono.get_theta()
        step_res = env_mono.step(theta_curr)
        dataset = step_res["dataset"]

        pol = compute_polarization_metrics(env_mono.population)
        drift = compute_parameter_drift(theta_curr, theta_prev_mono) if theta_prev_mono is not None else 0.0
        risk = compute_empirical_risk(model_mono, dataset)

        record = {
            "round": t,
            "setting": "monopoly",
            "market_share_A": 1.0,
            "market_share_B": 0.0,
            "inter_platform_drift": 0.0,
            "drift_A": drift,
            "drift_B": 0.0,
            "empirical_risk_A": risk,
            "empirical_risk_B": 0.0,
            **pol,
        }
        records.append(record)

        theta_prev_mono = theta_curr
        model_mono.fit(dataset)

    df = pd.DataFrame(records)
    csv_path = output_dir / "exp3_duopoly_polarization.csv"
    json_path = output_dir / "exp3_duopoly_polarization.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\n[Exp 3 Done] Saved {len(df)} records to {csv_path} and {json_path}")
    return df


if __name__ == "__main__":
    out = REPO_ROOT / "results"
    run_duopoly_experiment(out)
