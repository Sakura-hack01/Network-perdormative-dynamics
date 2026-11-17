"""Unified Command Line Interface for Networked Performative Dynamics.

Usage:
    python -m src.cli test
    python -m src.cli exp --id [1|2|3|4|all]
    python -m src.cli plot
    python -m src.cli simulate --epsilon 0.25 --lambda 0.3 --kappa 0.2 --algorithm rrm
    python -m src.cli serve --port 8000
"""

import argparse
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def cmd_test(args):
    """Executes the pytest test suite."""
    import subprocess
    print("=== Running PyTest Test Suite ===")
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "-p", "no:cacheprovider"]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    sys.exit(res.returncode)


def cmd_exp(args):
    """Executes experimental benchmarks."""
    exp_id = args.id
    if exp_id in ["1", "all"]:
        from experiments.exp1_contraction_breakdown import run_experiment
        run_experiment(REPO_ROOT / "results")
    if exp_id in ["2", "all"]:
        from experiments.exp2_phase_diagram import run_phase_sweep
        run_phase_sweep(REPO_ROOT / "results")
    if exp_id in ["3", "all"]:
        from experiments.exp3_duopoly_polarization import run_duopoly_experiment
        run_duopoly_experiment(REPO_ROOT / "results")
    if exp_id in ["4", "all"]:
        from experiments.exp4_ablation_exploration import run_ablation_experiment
        run_ablation_experiment(REPO_ROOT / "results")


def cmd_plot(args):
    """Generates 300-DPI publication figures."""
    from visualization.plot_figures import generate_all_figures
    generate_all_figures()


def cmd_simulate(args):
    """Runs on-the-fly custom dynamical simulation and prints metrics."""
    from src.env.network_population import NetworkPopulation
    from src.env.stateful_feedback import StatefulFeedbackEnv, ItemCatalog
    from src.algorithms.rrm import RepeatedRiskMinimization
    from src.algorithms.damped_rgd import DampedRGD
    from src.algorithms.static_erm import StaticERM
    from src.algorithms.causal_lookahead_po import CausalLookaheadPO
    from src.metrics.evaluators import compute_polarization_metrics

    print(f"\n=== Simulating Performative Dynamics ===")
    print(f"Params: epsilon={args.epsilon}, lambda={args.lambda_mem}, kappa={args.kappa}, alg={args.algorithm}")

    pop = NetworkPopulation(
        num_users=args.users,
        dimension=args.dimension,
        community_sep=0.6,
        seed=args.seed,
    )
    catalog = ItemCatalog(num_items=args.items, dimension=args.dimension, seed=args.seed)
    env = StatefulFeedbackEnv(
        population=pop,
        catalog=catalog,
        slate_size=args.slate_size,
        beta=args.beta,
        lambda_mem=args.lambda_mem,
        kappa_peer=args.kappa,
        epsilon_perf=args.epsilon,
        seed=args.seed,
    )

    algs = {
        "rrm": lambda: RepeatedRiskMinimization(args.items, args.dimension, l2_reg=args.l2_reg),
        "rgd": lambda: DampedRGD(args.items, args.dimension, l2_reg=args.l2_reg, lr=0.05),
        "static": lambda: StaticERM(args.items, args.dimension, l2_reg=args.l2_reg),
        "lookahead": lambda: CausalLookaheadPO(args.items, args.dimension, l2_reg=args.l2_reg),
    }

    model = algs.get(args.algorithm.lower(), algs["rrm"])()
    model.init_theta()

    print(f"\n{'Round':<6} | {'Drift ||dtheta||':<16} | {'Bimodality (b)':<15} | {'Comm Separation':<15} | {'Welfare':<10}")
    print("-" * 72)

    theta_prev = None
    for t in range(args.rounds):
        theta_curr = model.get_theta()
        step_res = env.step(theta_curr)
        pol = compute_polarization_metrics(pop)

        drift_val = 0.0
        if theta_prev is not None:
            import numpy as np
            drift_val = float(np.linalg.norm(theta_curr - theta_prev))

        welfare_val = step_res["welfare"]
        bimod_val = pol["bimodality_coefficient"]
        sep_val = pol["community_separation"]

        print(f"{t:<6} | {drift_val:<16.5f} | {bimod_val:<15.4f} | {sep_val:<15.4f} | {welfare_val:<10.4f}")

        theta_prev = theta_curr
        if args.algorithm.lower() == "lookahead":
            model.lookahead_step(env, step_res["dataset"])
        else:
            model.fit(step_res["dataset"])

    print("\nSimulation complete.\n")


def cmd_serve(args):
    """Starts the interactive Web Dashboard server."""
    from app import run_server
    run_server(port=args.port, host=args.host)


def main():
    parser = argparse.ArgumentParser(
        description="Networked Performative Dynamics Research CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Test
    p_test = subparsers.add_parser("test", help="Run pytest unit test suite")
    p_test.set_defaults(func=cmd_test)

    # Experiments
    p_exp = subparsers.add_parser("exp", help="Run experimental benchmarks")
    p_exp.add_argument("--id", choices=["1", "2", "3", "4", "all"], default="all", help="Experiment to run")
    p_exp.set_defaults(func=cmd_exp)

    # Plot
    p_plot = subparsers.add_parser("plot", help="Generate all 300-DPI publication figures")
    p_plot.set_defaults(func=cmd_plot)

    # Simulate
    p_sim = subparsers.add_parser("simulate", help="Run custom on-the-fly dynamical simulation")
    p_sim.add_argument("--epsilon", type=float, default=0.20, help="Performative sensitivity")
    p_sim.add_argument("--lambda_mem", type=float, default=0.30, help="Habit memory parameter")
    p_sim.add_argument("--kappa", type=float, default=0.25, help="Peer social coupling")
    p_sim.add_argument("--beta", type=float, default=2.0, help="Inverse choice temperature")
    p_sim.add_argument("--l2_reg", type=float, default=1.0, help="L2 regularization (gamma)")
    p_sim.add_argument("--algorithm", choices=["rrm", "rgd", "static", "lookahead"], default="rrm")
    p_sim.add_argument("--rounds", type=int, default=15, help="Number of interaction rounds")
    p_sim.add_argument("--users", type=int, default=600, help="Population size")
    p_sim.add_argument("--dimension", type=int, default=8, help="Preference dimension")
    p_sim.add_argument("--items", type=int, default=30, help="Catalog size")
    p_sim.add_argument("--slate_size", type=int, default=5, help="Slate recommendation size")
    p_sim.add_argument("--seed", type=int, default=42, help="Random seed")
    p_sim.set_defaults(func=cmd_simulate)

    # Serve
    p_srv = subparsers.add_parser("serve", help="Launch interactive Web Dashboard")
    p_srv.add_argument("--host", default="127.0.0.1", help="Host interface")
    p_srv.add_argument("--port", type=int, default=8000, help="HTTP server port")
    p_srv.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
