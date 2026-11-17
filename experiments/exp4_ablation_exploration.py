"""Experiment 4: Graph Topology and Algorithmic Mitigation Ablation.

Systematically ablates:
1. Network Topologies: SBM (clustered), Watts-Strogatz (small-world), Erdős-Rényi, and Disconnected.
2. Mitigation Strategies: Causal Lookahead PO with cluster-randomization vs Standard RRM vs Damped RGD vs Static ERM.
Evaluates long-term Social Welfare, Performative Risk, and Echo-Chamber Polarization.
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
from src.algorithms.static_erm import StaticERM
from src.algorithms.rrm import RepeatedRiskMinimization
from src.algorithms.damped_rgd import DampedRGD
from src.algorithms.causal_lookahead_po import CausalLookaheadPO
from src.metrics.evaluators import evaluate_round_metrics


def run_ablation_experiment(
    output_dir: Path,
    num_users: int = 800,
    dimension: int = 16,
    num_items: int = 40,
    slate_size: int = 5,
    beta: float = 2.0,
    l2_reg: float = 1.0,
    rounds: int = 25,
    seed: int = 42,
) -> pd.DataFrame:
    """Executes network topology and algorithm ablation suite."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []

    # 1. Topology Ablation under standard RRM
    print("=== Running Exp 4: Topology Ablation (RRM) ===")
    topologies = ["sbm", "watts_strogatz", "erdos_renyi", "empty"]

    catalog = ItemCatalog(num_items, dimension, seed=seed, community_sep=0.6)

    for topo in topologies:
        print(f"  --> Testing Graph Topology: {topo}")
        pop = NetworkPopulation(
            num_users=num_users,
            dimension=dimension,
            num_communities=2,
            community_sep=0.6,
            anchor_noise_std=0.15,
            graph_type=topo,
            p_in=0.08,
            p_out=0.005,
            ws_k=10,
            ws_p=0.1,
            er_p=0.04,
            seed=seed,
        )
        env = StatefulFeedbackEnv(
            population=pop,
            catalog=catalog,
            slate_size=slate_size,
            beta=beta,
            tau_0=0.0,
            lambda_mem=0.30,
            kappa_peer=0.25 if topo != "empty" else 0.0,
            epsilon_perf=0.25,
            transition_noise_std=0.015,
            seed=seed,
        )

        model = RepeatedRiskMinimization(num_items, dimension, l2_reg=l2_reg, max_iter=25)
        model.init_theta()
        theta_prev = None

        for t in range(rounds):
            theta_curr = model.get_theta()
            step_res = env.step(theta_curr)
            dataset = step_res["dataset"]

            metrics = evaluate_round_metrics(
                round_idx=t,
                model=model,
                env=env,
                dataset=dataset,
                step_info=step_res,
                theta_prev=theta_prev,
            )
            metrics["ablation_type"] = "topology"
            metrics["graph_topology"] = topo
            metrics["algorithm"] = "RRM"
            records.append(metrics)

            theta_prev = theta_curr
            model.fit(dataset)

    # 2. Algorithm Mitigation Ablation on Clustered SBM Network
    print("\n=== Running Exp 4: Algorithm Mitigation Ablation (Clustered SBM) ===")
    algorithms = {
        "Static_ERM": lambda: StaticERM(num_items, dimension, l2_reg=l2_reg),
        "RRM": lambda: RepeatedRiskMinimization(num_items, dimension, l2_reg=l2_reg, max_iter=25),
        "Damped_RGD": lambda: DampedRGD(num_items, dimension, l2_reg=l2_reg, lr=0.05),
        "Causal_Lookahead_PO": lambda: CausalLookaheadPO(
            num_items, dimension, l2_reg=l2_reg, lr=0.05, delta=0.05, n_clusters=8, div_weight=0.08
        ),
    }

    for alg_name, alg_fn in algorithms.items():
        print(f"  --> Testing Mitigation Algorithm: {alg_name}")
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
        env = StatefulFeedbackEnv(
            population=pop,
            catalog=catalog,
            slate_size=slate_size,
            beta=beta,
            tau_0=0.0,
            lambda_mem=0.30,
            kappa_peer=0.25,
            epsilon_perf=0.30,
            transition_noise_std=0.015,
            seed=seed,
        )

        model = alg_fn()
        model.init_theta()
        theta_prev = None

        for t in range(rounds):
            theta_curr = model.get_theta()
            step_res = env.step(theta_curr)
            dataset = step_res["dataset"]

            metrics = evaluate_round_metrics(
                round_idx=t,
                model=model,
                env=env,
                dataset=dataset,
                step_info=step_res,
                theta_prev=theta_prev,
            )
            metrics["ablation_type"] = "algorithm"
            metrics["graph_topology"] = "sbm"
            metrics["algorithm"] = alg_name
            records.append(metrics)

            theta_prev = theta_curr
            if alg_name == "Causal_Lookahead_PO":
                model.lookahead_step(env, dataset)
            else:
                model.fit(dataset)

    df = pd.DataFrame(records)
    csv_path = output_dir / "exp4_ablation_exploration.csv"
    json_path = output_dir / "exp4_ablation_exploration.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\n[Exp 4 Done] Saved {len(df)} records to {csv_path} and {json_path}")
    return df


if __name__ == "__main__":
    out = REPO_ROOT / "results"
    run_ablation_experiment(out)
