"""Experiment 2: Dynamical Phase Diagram and Bifurcation Analysis.

Investigates echo-chamber bifurcations and phase boundaries in the 2D dynamical system (d=2)
across individual habit memory lambda in [0, 0.8] and social peer coupling kappa in [0, 0.8].
Evaluates Sarle's Bimodality Coefficient, Community Separation, Preference Dispersion,
and parameter stability to map where echo chambers emerge.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.env.network_population import NetworkPopulation
from src.env.stateful_feedback import StatefulFeedbackEnv, ItemCatalog
from src.algorithms.rrm import RepeatedRiskMinimization
from src.metrics.evaluators import evaluate_round_metrics


def run_phase_sweep(
    output_dir: Path,
    num_users: int = 600,
    dimension: int = 2,
    num_items: int = 30,
    slate_size: int = 4,
    beta: float = 2.5,
    l2_reg: float = 1.0,
    rounds: int = 25,
    seed: int = 42,
) -> pd.DataFrame:
    """Executes a 2D parameter grid sweep over (lambda, kappa) and (epsilon, kappa)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []

    # Grid 1: lambda (memory) vs kappa (peer influence) at fixed epsilon = 0.25
    lambda_grid = [0.0, 0.15, 0.30, 0.45, 0.60]
    kappa_grid = [0.0, 0.10, 0.20, 0.30, 0.40]
    fixed_eps = 0.25

    print(f"=== Running Exp 2: 2D Phase Diagram Sweep (Memory lambda vs Peer kappa) ===")
    for lam in lambda_grid:
        for kap in kappa_grid:
            # Physical stability constraint: lambda + kappa < 1.0
            if lam + kap >= 0.95:
                continue

            print(f"  --> Simulating grid point: lambda={lam:.2f}, kappa={kap:.2f}, eps={fixed_eps:.2f}")

            pop = NetworkPopulation(
                num_users=num_users,
                dimension=dimension,
                num_communities=2,
                community_sep=0.5,
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
                community_sep=0.5,
            )
            env = StatefulFeedbackEnv(
                population=pop,
                catalog=catalog,
                slate_size=slate_size,
                beta=beta,
                tau_0=0.0,
                lambda_mem=lam,
                kappa_peer=kap,
                epsilon_perf=fixed_eps,
                transition_noise_std=0.01,
                seed=seed,
            )

            model = RepeatedRiskMinimization(
                num_items=num_items,
                dimension=dimension,
                l2_reg=l2_reg,
                max_iter=25,
            )
            model.init_theta()

            theta_prev = None
            step_info = None
            dataset = None

            for t in range(rounds):
                theta_curr = model.get_theta()
                step_info = env.step(theta_curr)
                dataset = step_info["dataset"]

                if t == rounds - 1:
                    metrics = evaluate_round_metrics(
                        round_idx=t,
                        model=model,
                        env=env,
                        dataset=dataset,
                        step_info=step_info,
                        theta_prev=theta_prev,
                    )
                    metrics["lambda_mem"] = lam
                    metrics["kappa_peer"] = kap
                    metrics["epsilon_perf"] = fixed_eps
                    metrics["sweep_type"] = "lambda_vs_kappa"
                    # Classify phase: Echo-chamber if bimodality > 0.555 (Sarle's threshold)
                    metrics["is_bimodal_echo_chamber"] = bool(metrics["bimodality_coefficient"] > 0.555)
                    records.append(metrics)

                theta_prev = theta_curr
                model.fit(dataset)

    # Grid 2: epsilon (performativity) vs kappa (peer influence) at fixed lambda = 0.30
    epsilon_grid = [0.0, 0.10, 0.20, 0.35, 0.50]
    kappa_grid_2 = [0.0, 0.10, 0.20, 0.30, 0.40]
    fixed_lam = 0.30

    print(f"\n=== Running Exp 2: 2D Phase Diagram Sweep (Performativity eps vs Peer kappa) ===")
    for eps in epsilon_grid:
        for kap in kappa_grid_2:
            print(f"  --> Simulating grid point: eps={eps:.2f}, kappa={kap:.2f}, lambda={fixed_lam:.2f}")

            pop = NetworkPopulation(
                num_users=num_users,
                dimension=dimension,
                num_communities=2,
                community_sep=0.5,
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
                community_sep=0.5,
            )
            env = StatefulFeedbackEnv(
                population=pop,
                catalog=catalog,
                slate_size=slate_size,
                beta=beta,
                tau_0=0.0,
                lambda_mem=fixed_lam,
                kappa_peer=kap,
                epsilon_perf=eps,
                transition_noise_std=0.01,
                seed=seed,
            )

            model = RepeatedRiskMinimization(
                num_items=num_items,
                dimension=dimension,
                l2_reg=l2_reg,
                max_iter=25,
            )
            model.init_theta()

            theta_prev = None
            step_info = None
            dataset = None

            for t in range(rounds):
                theta_curr = model.get_theta()
                step_info = env.step(theta_curr)
                dataset = step_info["dataset"]

                if t == rounds - 1:
                    metrics = evaluate_round_metrics(
                        round_idx=t,
                        model=model,
                        env=env,
                        dataset=dataset,
                        step_info=step_info,
                        theta_prev=theta_prev,
                    )
                    metrics["lambda_mem"] = fixed_lam
                    metrics["kappa_peer"] = kap
                    metrics["epsilon_perf"] = eps
                    metrics["sweep_type"] = "epsilon_vs_kappa"
                    metrics["is_bimodal_echo_chamber"] = bool(metrics["bimodality_coefficient"] > 0.555)
                    records.append(metrics)

                theta_prev = theta_curr
                model.fit(dataset)

    df = pd.DataFrame(records)
    csv_path = output_dir / "exp2_phase_diagram.csv"
    json_path = output_dir / "exp2_phase_diagram.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\n[Exp 2 Done] Saved {len(df)} phase points to {csv_path} and {json_path}")
    return df


if __name__ == "__main__":
    out = REPO_ROOT / "results"
    run_phase_sweep(out)
