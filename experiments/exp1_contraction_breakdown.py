"""Experiment 1: Contraction Breakdown in Performative Prediction.

Investigates the breakdown of standard contraction guarantees (Perdomo et al., 2020)
as performative sensitivity epsilon increases past the theoretical threshold gamma / beta.
Evaluates parameter drift ||theta_{t+1} - theta_t||_F across rounds t, comparing
Static ERM, RRM, Damped RGD, and Causal Lookahead PO.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.env.network_population import NetworkPopulation
from src.env.stateful_feedback import StatefulFeedbackEnv, ItemCatalog
from src.algorithms.static_erm import StaticERM
from src.algorithms.rrm import RepeatedRiskMinimization
from src.algorithms.damped_rgd import DampedRGD
from src.algorithms.causal_lookahead_po import CausalLookaheadPO
from src.metrics.evaluators import evaluate_round_metrics


def run_experiment(
    output_dir: Path,
    num_users: int = 1000,
    dimension: int = 16,
    num_items: int = 50,
    slate_size: int = 5,
    beta: float = 2.0,
    l2_reg: float = 1.0,
    rounds: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Executes Experiment 1 sweep across epsilon and algorithms."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict] = []

    # 1. Sweep epsilon for RRM to observe the contraction-to-divergence transition
    epsilon_sweep = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    print(f"=== Running Exp 1: Epsilon Contraction Sweep for RRM ===")

    for eps in epsilon_sweep:
        print(f"  --> Simulating RRM with epsilon = {eps:.2f}")
        # Build clean environment
        pop = NetworkPopulation(
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
        env = StatefulFeedbackEnv(
            population=pop,
            catalog=catalog,
            slate_size=slate_size,
            beta=beta,
            tau_0=0.0,
            lambda_mem=0.0,   # Memoryless baseline for pure Perdomo contraction comparison
            kappa_peer=0.0,   # No peer interference for clean contraction boundary
            epsilon_perf=eps,
            transition_noise_std=0.01,
            seed=seed,
        )

        model = RepeatedRiskMinimization(
            num_items=num_items,
            dimension=dimension,
            l2_reg=l2_reg,
            max_iter=30,
        )
        model.init_theta()

        theta_prev = None

        for t in range(rounds):
            theta_curr = model.get_theta()
            step_info = env.step(theta_curr)
            dataset = step_info["dataset"]

            metrics = evaluate_round_metrics(
                round_idx=t,
                model=model,
                env=env,
                dataset=dataset,
                step_info=step_info,
                theta_prev=theta_prev,
            )
            metrics["epsilon"] = eps
            metrics["algorithm"] = "RRM"
            metrics["regime"] = "epsilon_sweep"
            results.append(metrics)

            theta_prev = theta_curr
            model.fit(dataset)

    # 2. Compare Algorithms at Performative Regime (epsilon = 0.35, lambda = 0.3, kappa = 0.2)
    print("\n=== Running Exp 1: Algorithm Comparison under Stateful Interference ===")
    algorithms = {
        "Static_ERM": lambda: StaticERM(num_items, dimension, l2_reg=l2_reg),
        "RRM": lambda: RepeatedRiskMinimization(num_items, dimension, l2_reg=l2_reg, max_iter=30),
        "Damped_RGD": lambda: DampedRGD(num_items, dimension, l2_reg=l2_reg, lr=0.05),
        "Causal_Lookahead_PO": lambda: CausalLookaheadPO(
            num_items, dimension, l2_reg=l2_reg, lr=0.05, delta=0.05, n_clusters=8, div_weight=0.05
        ),
    }

    test_eps = 0.35
    for alg_name, alg_fn in algorithms.items():
        print(f"  --> Simulating {alg_name} (eps={test_eps})")
        pop = NetworkPopulation(
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
        env = StatefulFeedbackEnv(
            population=pop,
            catalog=catalog,
            slate_size=slate_size,
            beta=beta,
            tau_0=0.0,
            lambda_mem=0.30,
            kappa_peer=0.25,
            epsilon_perf=test_eps,
            transition_noise_std=0.015,
            seed=seed,
        )

        model = alg_fn()
        model.init_theta()
        theta_prev = None

        for t in range(rounds):
            theta_curr = model.get_theta()
            step_info = env.step(theta_curr)
            dataset = step_info["dataset"]

            metrics = evaluate_round_metrics(
                round_idx=t,
                model=model,
                env=env,
                dataset=dataset,
                step_info=step_info,
                theta_prev=theta_prev,
            )
            metrics["epsilon"] = test_eps
            metrics["algorithm"] = alg_name
            metrics["regime"] = "algorithm_comparison"
            results.append(metrics)

            theta_prev = theta_curr
            if alg_name == "Causal_Lookahead_PO":
                model.lookahead_step(env, dataset)
            else:
                model.fit(dataset)

    df = pd.DataFrame(results)

    # Save CSV and JSON
    csv_path = output_dir / "exp1_contraction_breakdown.csv"
    json_path = output_dir / "exp1_contraction_breakdown.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Exp 1 Done] Saved {len(df)} records to {csv_path} and {json_path}")
    return df


if __name__ == "__main__":
    out = REPO_ROOT / "results"
    run_experiment(out)
