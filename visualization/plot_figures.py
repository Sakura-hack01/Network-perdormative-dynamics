"""Publication-grade visualization script for Networked Performative Dynamics.

Generates 300-DPI publication figures:
- Figure 1: Contraction Breakdown and Rate Curves (Exp 1).
- Figure 2: Phase Diagram, Bifurcation Heatmaps, and 2D Streamlines (Exp 2).
- Figure 3: Duopoly Competition Dynamics and Market Separation (Exp 3).
- Figure 4: Network Topology and Algorithmic Mitigation Ablation (Exp 4).
"""

import os
import sys
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

# Set safe writable matplotlib config dir
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Publication aesthetic settings
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "figure.autolayout": False,
})


def plot_figure_1(results_dir: Path, output_dir: Path) -> None:
    """Figure 1: Contraction Breakdown and Convergence Rate Curves."""
    csv_file = results_dir / "exp1_contraction_breakdown.csv"
    if not csv_file.exists():
        print(f"[Warning] {csv_file} not found. Skipping Fig 1.")
        return

    df = pd.read_csv(csv_file)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot A: Epsilon sweep contraction curves (semi-log)
    df_eps = df[df["regime"] == "epsilon_sweep"]
    palette = sns.color_palette("viridis", n_colors=df_eps["epsilon"].nunique())

    for idx, (eps, grp) in enumerate(df_eps.groupby("epsilon")):
        grp_sorted = grp.sort_values("round")
        # Shift round by 1 for drift
        rounds = grp_sorted["round"].values[1:]
        drift = grp_sorted["parameter_drift"].values[1:]
        drift = np.maximum(drift, 1e-8)
        axes[0].semilogy(
            rounds,
            drift,
            marker="o",
            label=rf"$\epsilon = {eps:.2f}$",
            color=palette[idx],
        )

    axes[0].set_xlabel("Rounds $t$")
    axes[0].set_ylabel(r"Parameter Drift $\|\theta_{t+1} - \theta_t\|_F$ (log-scale)")
    axes[0].set_title(r"(a) Contraction Rate Breakdown vs. $\epsilon$ (RRM)")
    axes[0].legend(loc="upper right", framealpha=0.9)

    # Subplot B: Algorithm comparison
    df_alg = df[df["regime"] == "algorithm_comparison"]
    alg_palette = sns.color_palette("Set2", n_colors=df_alg["algorithm"].nunique())

    for idx, (alg, grp) in enumerate(df_alg.groupby("algorithm")):
        grp_sorted = grp.sort_values("round")
        axes[1].plot(
            grp_sorted["round"].values,
            grp_sorted["performative_risk"].values,
            marker="s",
            label=alg.replace("_", " "),
            color=alg_palette[idx],
        )

    axes[1].set_xlabel("Rounds $t$")
    axes[1].set_ylabel(r"Performative Risk $\mathrm{PR}(\theta_t)$")
    axes[1].set_title(r"(b) Performative Risk under Stateful Interference")
    axes[1].legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_contraction_breakdown.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig1_contraction_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved Fig 1 to figures/fig1_contraction_breakdown.png")


def plot_figure_2(results_dir: Path, output_dir: Path) -> None:
    """Figure 2: 2D Dynamical Phase Diagram and Echo-Chamber Bifurcations."""
    csv_file = results_dir / "exp2_phase_diagram.csv"
    if not csv_file.exists():
        print(f"[Warning] {csv_file} not found. Skipping Fig 2.")
        return

    df = pd.read_csv(csv_file)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Subplot A: Heatmap of Sarle's Bimodality Coefficient (lambda vs kappa)
    df_lam = df[df["sweep_type"] == "lambda_vs_kappa"]
    if not df_lam.empty:
        pivot_bimod = df_lam.pivot(
            index="lambda_mem", columns="kappa_peer", values="bimodality_coefficient"
        )
        sns.heatmap(
            pivot_bimod,
            annot=True,
            fmt=".3f",
            cmap="mako",
            cbar_kws={"label": "Sarle's Bimodality $b$"},
            ax=axes[0],
        )
        axes[0].invert_yaxis()
        axes[0].set_xlabel(r"Peer Coupling $\kappa$ (Social Influence)")
        axes[0].set_ylabel(r"Habit Memory $\lambda$ (Path Dependence)")
        axes[0].set_title(r"(a) Echo-Chamber Bifurcation ($b > 0.555$)")

    # Subplot B: Heatmap of Community Separation (epsilon vs kappa)
    df_eps = df[df["sweep_type"] == "epsilon_vs_kappa"]
    if not df_eps.empty:
        pivot_sep = df_eps.pivot(
            index="epsilon_perf", columns="kappa_peer", values="community_separation"
        )
        sns.heatmap(
            pivot_sep,
            annot=True,
            fmt=".3f",
            cmap="rocket_r",
            cbar_kws={"label": r"Community Separation $\|\bar{u}_0 - \bar{u}_1\|_2$"},
            ax=axes[1],
        )
        axes[1].invert_yaxis()
        axes[1].set_xlabel(r"Peer Coupling $\kappa$ (Social Influence)")
        axes[1].set_ylabel(r"Performative Sensitivity $\epsilon$")
        axes[1].set_title(r"(b) Preference Polarization Across Communities")

    plt.tight_layout()
    fig.savefig(output_dir / "fig2_phase_diagram.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig2_phase_diagram.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved Fig 2 to figures/fig2_phase_diagram.png")


def plot_figure_3(results_dir: Path, output_dir: Path) -> None:
    """Figure 3: Duopoly Competition Dynamics and Polarization Acceleration."""
    csv_file = results_dir / "exp3_duopoly_polarization.csv"
    if not csv_file.exists():
        print(f"[Warning] {csv_file} not found. Skipping Fig 3.")
        return

    df = pd.read_csv(csv_file)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    df_duo = df[df["setting"] == "duopoly"].sort_values("round")
    df_mono = df[df["setting"] == "monopoly"].sort_values("round")

    # Subplot A: Market share dynamics
    axes[0].plot(df_duo["round"], df_duo["market_share_A"], label="Platform A", color="#1f77b4")
    axes[0].plot(df_duo["round"], df_duo["market_share_B"], label="Platform B", color="#ff7f0e", linestyle="--")
    axes[0].set_xlabel("Rounds $t$")
    axes[0].set_ylabel("Market Share")
    axes[0].set_title("(a) Duopoly Platform Market Shares")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(loc="center right")

    # Subplot B: Inter-platform model divergence
    axes[1].plot(
        df_duo["round"],
        df_duo["inter_platform_drift"],
        color="#2ca02c",
        marker="o",
        markevery=3,
        label=r"$\|\theta_A - \theta_B\|_F$",
    )
    axes[1].set_xlabel("Rounds $t$")
    axes[1].set_ylabel(r"Model Parameter Divergence $\|\theta_A - \theta_B\|_F$")
    axes[1].set_title("(b) Platform Strategy Divergence")
    axes[1].legend(loc="upper left")

    # Subplot C: Community separation: Duopoly vs Monopoly
    axes[2].plot(
        df_duo["round"],
        df_duo["community_separation"],
        label="Duopoly Competition",
        color="#d62728",
        marker="^",
        markevery=3,
    )
    axes[2].plot(
        df_mono["round"],
        df_mono["community_separation"],
        label="Monopoly Baseline",
        color="#7f7f7f",
        linestyle="--",
        marker="s",
        markevery=3,
    )
    axes[2].set_xlabel("Rounds $t$")
    axes[2].set_ylabel(r"Community Separation $\|\bar{u}_0 - \bar{u}_1\|_2$")
    axes[2].set_title("(c) Polarization Acceleration")
    axes[2].legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(output_dir / "fig3_duopoly_polarization.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig3_duopoly_polarization.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved Fig 3 to figures/fig3_duopoly_polarization.png")


def plot_figure_4(results_dir: Path, output_dir: Path) -> None:
    """Figure 4: Network Topology and Algorithmic Mitigation Ablation."""
    csv_file = results_dir / "exp4_ablation_exploration.csv"
    if not csv_file.exists():
        print(f"[Warning] {csv_file} not found. Skipping Fig 4.")
        return

    df = pd.read_csv(csv_file)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot A: Topology ablation on community separation
    df_topo = df[df["ablation_type"] == "topology"]
    topo_palette = sns.color_palette("tab10", n_colors=df_topo["graph_topology"].nunique())

    for idx, (topo, grp) in enumerate(df_topo.groupby("graph_topology")):
        grp_s = grp.sort_values("round")
        label_map = {
            "sbm": "SBM (Clustered)",
            "watts_strogatz": "Watts-Strogatz (Small-World)",
            "erdos_renyi": "Erdős-Rényi (Random)",
            "empty": "No Network Interference",
        }
        axes[0].plot(
            grp_s["round"],
            grp_s["community_separation"],
            label=label_map.get(topo, topo),
            color=topo_palette[idx],
            marker="o",
            markevery=3,
        )

    axes[0].set_xlabel("Rounds $t$")
    axes[0].set_ylabel(r"Community Separation $\|\bar{u}_0 - \bar{u}_1\|_2$")
    axes[0].set_title("(a) Echo-Chamber Severity by Graph Topology")
    axes[0].legend(loc="upper left")

    # Subplot B: Algorithmic mitigation on social welfare
    df_alg = df[df["ablation_type"] == "algorithm"]
    alg_palette = sns.color_palette("Dark2", n_colors=df_alg["algorithm"].nunique())

    for idx, (alg, grp) in enumerate(df_alg.groupby("algorithm")):
        grp_s = grp.sort_values("round")
        axes[1].plot(
            grp_s["round"],
            grp_s["social_welfare"],
            label=alg.replace("_", " "),
            color=alg_palette[idx],
            marker="s",
            markevery=3,
        )

    axes[1].set_xlabel("Rounds $t$")
    axes[1].set_ylabel("Social Welfare (Realized Utility)")
    axes[1].set_title("(b) Long-Term Welfare via Causal Lookahead")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_ablation_mitigation.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig4_ablation_mitigation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved Fig 4 to figures/fig4_ablation_mitigation.png")


def generate_all_figures():
    """Generates all publication-grade figures."""
    results_dir = REPO_ROOT / "results"
    output_dir = REPO_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures from {results_dir} into {output_dir}...")
    plot_figure_1(results_dir, output_dir)
    plot_figure_2(results_dir, output_dir)
    plot_figure_3(results_dir, output_dir)
    plot_figure_4(results_dir, output_dir)
    print("All figures successfully generated.")


if __name__ == "__main__":
    generate_all_figures()
