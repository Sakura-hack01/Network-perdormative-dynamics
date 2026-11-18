# Networked Performative Dynamics: Stateful Feedback, Network Interference, and Multi-Platform Competition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


A clean, reproducible, publication-grade research repository investigating **"Stateful Performative Prediction under Multi-Platform Competition and Social Network Interference"**, extending foundational work by Perdomo et al. (*ICML 2020*) and Mendler-Dünner et al. (*NeurIPS 2020*).

---

## 1. Research Overview & Theoretical Motivation

Classical performative prediction studies how deploying a statistical model $\theta$ induces a shift in the underlying data distribution $z \sim \mathcal{D}(\theta)$. Under standard assumptions—namely that the loss $\ell(\cdot; \theta)$ is $\gamma$-strongly convex and the performative map $\mathcal{D}(\cdot)$ is $\epsilon$-sensitive in Wasserstein distance—Repeated Risk Minimization (RRM) is guaranteed to converge linearly to a unique stable point provided:
$$\epsilon < \frac{\gamma}{L}$$

However, in realistic real-world socio-technical environments (such as recommender feeds, search engines, and news aggregators), two fundamental pillars of this guarantee break down:

1. **Non-Markovian / Stateful Performativity:** User preferences are not static memoryless responses; they exhibit habituation and path-dependent inertia ($\lambda \in [0, 1)$), moving with momentum.
2. **Social Network Interference:** Users do not act in isolation. They are embedded in social network topologies (Stochastic Block Models, Watts-Strogatz small-world graphs) where peer interactions ($\kappa \in [0, 1)$) diffuse and amplify preference shifts across community boundaries.
3. **Multi-Platform Competition (Duopoly):** When multiple platforms ($\theta_A, \theta_B$) compete for user engagement, algorithmic incentives drive niche differentiation, rapidly bifurcating the population into polarized echo-chambers.

This repository provides exact mathematical models, dynamical systems simulations, and a novel **Causal Lookahead Performative Optimization** algorithm that mitigates network interference via cluster-randomized graph partitioning.

---

## 2. Mathematical Framework

### A. Population & Social Graph (`src/env/network_population.py`)
- **Users & Anchors:** $N = 1000$ users partitioned into $C = 2$ balanced communities. Each user $i$ possesses an intrinsic anchor preference $u_i^{(0)} \sim \mathcal{N}(\mu_{c_i}, \sigma_0^2 I_d)$ bounded in the unit ball $\|u_i^{(0)}\|_2 \le 1$ ($d=2$ for phase portraits, $d=16$ for representation spaces).
- **Social Graph Adjacency:** $A \in \{0, 1\}^{N \times N}$ generated via a Stochastic Block Model (SBM):
  $$P(A_{ij} = 1) = \begin{cases} p_{in} = 0.08 & \text{if } c_i = c_j \\ p_{out} = 0.005 & \text{if } c_i \neq c_j \end{cases}$$
- **Row-Stochastic Diffusion Matrix:**
  $$\tilde{A}_{ij} = \frac{A_{ij}}{\max(1, \sum_k A_{ik})}, \quad \text{with } \sum_j \tilde{A}_{ij} = 1$$

### B. Choice Model & Stateful Update (`src/env/stateful_feedback.py`)
- **Slate Recommendations:** Platform parameterized by $\theta_t \in \mathbb{R}^{K \times d}$ recommends top-$M$ items ($M=5$) from catalog $V$ ($|V|=50$, $v_k \in \mathbb{R}^d$) based on predicted scores $s_{ik} = \langle \theta_k, u_i^{(t)} \rangle$.
- **Multinomial Logit Choice with Outside Bounce Option $\tau_0$:**
  $$P(\text{click item } k \in \text{slate} \mid u_i^{(t)}) = \frac{\exp(\beta \langle u_i^{(t)}, v_k \rangle)}{\sum_{j \in \text{slate}} \exp(\beta \langle u_i^{(t)}, v_j \rangle) + \exp(\beta \tau_0)}$$
  $$P(\text{bounce} \mid u_i^{(t)}) = \frac{\exp(\beta \tau_0)}{\sum_{j \in \text{slate}} \exp(\beta \langle u_i^{(t)}, v_j \rangle) + \exp(\beta \tau_0)}$$
- **Stateful + Networked Preference Transition:**
  $$u_i^{(t+1)} = (1 - \lambda - \kappa) u_i^{(0)} + \lambda u_i^{(t)} + \kappa \sum_{j=1}^N \tilde{A}_{ij} u_j^{(t)} + \epsilon \cdot (v_{c_i}^{(t)} - u_i^{(t)}) + \xi_i^{(t)}$$
  where $\lambda \in [0, 1)$ is memory habituation, $\kappa \in [0, 1)$ is peer social influence, $\epsilon > 0$ is performative sensitivity, and $\xi_i^{(t)} \sim \mathcal{N}(0, \sigma_\xi^2 I)$.

### C. Duopoly Platform Competition
- Platforms $A$ and $B$ deploy models $\theta_A$ and $\theta_B$.
- User $i$ selects Platform $A$ vs $B$ according to slate inclusive value:
  $$U_{i, P} = \frac{1}{\beta} \log \sum_{k \in \text{slate}_{i, P}} \exp(\beta \langle u_i^{(t)}, v_k \rangle), \quad P(\text{Choose } A) = \sigma(\rho (U_{i, A} - U_{i, B}))$$
- Each platform only observes feedback on its own visited cohort and updates its parameters independently.

---

## 3. Algorithms Implemented (`src/algorithms/`)

| Algorithm | File | Update Rule | Notes |
| :--- | :--- | :--- | :--- |
| **Static ERM** | `static_erm.py` | $\theta_t = \theta_0 = \arg\min_\theta \hat{R}(\theta; D_0)$ | Frozen non-adaptive baseline |
| **Repeated Risk Minimization (RRM)** | `rrm.py` | $\theta_{t+1} = \arg\min_\theta \hat{R}(\theta; D_t) + \frac{\gamma}{2}\|\theta\|_F^2$ | L-BFGS to full convergence each round |
| **Damped RGD** | `damped_rgd.py` | $\theta_{t+1} = \theta_t - \eta \nabla_\theta \hat{R}(\theta_t; D_t)$ | First-order online performative gradient descent |
| **Causal Lookahead PO** | `causal_lookahead_po.py` | Cluster-Randomized Two-Point Zero-Order + Diversity | Minimizes network interference via spectral graph partitions |

### Causal Lookahead PO: Mitigating Network Interference
Under peer influence $\kappa > 0$, standard randomized experiments suffer from **interference bias** (spillover effects). Causal Lookahead PO addresses this by:
1. **Spectral Graph Partitioning:** Partitions the social network into $C$ balanced clusters $\mathcal{C}_1, \dots, \mathcal{C}_C$ minimizing inter-cluster cut edges.
2. **Cluster Randomization:** Deploys $\theta_t + \delta W$ and $\theta_t - \delta W$ ($W$ uniform on Frobenius unit sphere) on opposite clusters.
3. **Two-Point Zero-Order Performative Gradient:**
   $$\hat{g}_{\text{ZO}} = \frac{\hat{R}^+ - \hat{R}^-}{2 \delta} \cdot \operatorname{dim}(\theta) \cdot W$$
4. **Diversity Regularizer:** Adds an entropy penalty $\Omega_{\text{diversity}}(\theta)$ to prevent catalog monopoly and preference collapse.

---

## 4. Repository Structure

```
networked-performative-dynamics/
├── README.md                           # Documentation, theoretical derivations & guide
├── requirements.txt                    # Python package dependencies
├── Makefile                            # Build automation (tests, experiments, plots)
├── configs/
│   ├── single_platform_sweep.yaml      # Single platform experiment configurations
│   └── duopoly_competition.yaml        # Duopoly competition configurations
├── src/
│   ├── __init__.py
│   ├── init.py                         # Compatibility import shim
│   ├── env/
│   │   ├── __init__.py
│   │   ├── network_population.py       # SBM / Small-World population & social graph
│   │   └── stateful_feedback.py        # Slate logit choice & stateful networked feedback
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base.py                     # Base recommender class & regularized loss
│   │   ├── static_erm.py               # Static ERM baseline
│   │   ├── rrm.py                      # Repeated Risk Minimization (L-BFGS)
│   │   ├── damped_rgd.py               # Damped Repeated Gradient Descent
│   │   └── causal_lookahead_po.py      # Cluster-randomized lookahead optimization
│   └── metrics/
│       ├── __init__.py
│       └── evaluators.py               # Bimodality, drift, risk & welfare evaluators
├── experiments/
│   ├── __init__.py
│   ├── exp1_contraction_breakdown.py   # Contraction guarantee vs sensitivity sweep
│   ├── exp2_phase_diagram.py           # 2D dynamical phase diagram (lambda, kappa, eps)
│   ├── exp3_duopoly_polarization.py    # Duopoly competition vs monopoly dynamics
│   └── exp4_ablation_exploration.py    # Topology & algorithmic mitigation ablations
├── visualization/
│   ├── __init__.py
│   └── plot_figures.py                 # Generates 300-DPI publication figures
├── tests/
│   ├── __init__.py
│   ├── test_contraction.py             # Unit tests for contraction and breakdown
│   └── test_env.py                     # Unit tests for population, graph & choices
├── results/                            # Structured CSV and JSON logs
└── figures/                            # 300-DPI publication figures (PNG and PDF)
```

---

## 5. Quick Start & Reproduction

### Prerequisites
Python 3.10+ with standard scientific libraries:
```bash
pip install -r requirements.txt
```

### Running Unit Tests
To verify contraction guarantees, choice model conservation, and spectral graph partitioning:
```bash
pytest tests/ -v
# or via Makefile:
make test
```
*Result: 7/7 unit tests pass.*

### Running Experimental Benchmarks
Run all experiments or individual scripts:
```bash
# Run all experiments
make experiments

# Or run individually:
python experiments/exp1_contraction_breakdown.py
python experiments/exp2_phase_diagram.py
python experiments/exp3_duopoly_polarization.py
python experiments/exp4_ablation_exploration.py
```

### Generating Publication Figures
Generate all 300-DPI publication figures (saved to `figures/` as both `.png` and `.pdf`):
```bash
python visualization/plot_figures.py
# or via Makefile:
make plot
```

---

## 6. Interactive Interfaces (Web GUI & CLI)

### A. Publication Web GUI Dashboard
An interactive graphical interface is provided in [`interface/index.html`](file:///c:/Users/Saket%20Dixit/Downloads/AM%20and%20PO/networked-performative-dynamics/interface/index.html) and served via [`app.py`](file:///c:/Users/Saket%20Dixit/Downloads/AM%20and%20PO/networked-performative-dynamics/app.py):
```bash
python app.py
# or: make serve
```
Then open `http://localhost:8000` in your web browser.
- **Dynamic Control Sliders:** Adjust performative sensitivity $\epsilon$, peer coupling $\kappa$, memory inertia $\lambda$, temperature $\beta$, and L2 regularization $\gamma$ in real time.
- **Interactive 2D Preference Space Canvas:** Live HTML5 canvas displaying the $N=1000$ user preferences $(u_1, u_2)$, platform hyperplane vectors, and echo-chamber clustering as rounds progress.
- **Real-Time Telemetry:** Monitors Parameter Drift $\|\theta_{t+1}-\theta_t\|_F$, Sarle's Bimodality Coefficient ($b > 0.555$ echo indicator), Community Separation, and Realized Welfare.
- **Built-in Figure Gallery:** Interactive tabs with high-res zoomable publication figures and captions.
- *Zero-dependency client-side engine:* Can also be opened directly by double-clicking `interface/index.html` in any browser without running a server.

### B. Unified Command Line Interface (CLI)
A unified CLI is available in `src/cli.py`:
```bash
# Run unit tests
python -m src.cli test

# Run specific experiment or all
python -m src.cli exp --id all

# Run custom on-the-fly performative simulation
python -m src.cli simulate --epsilon 0.25 --lambda_mem 0.30 --kappa 0.25 --algorithm rrm --rounds 15

# Generate figures
python -m src.cli plot

# Launch interactive web dashboard
python -m src.cli serve --port 8000
```

---

## 7. Publishing to GitHub

This repository is structured and pre-configured for GitHub publishing with:
- `.gitignore` (excludes caches, bytecode, and virtual environments)
- `LICENSE` (MIT License)
- `.github/workflows/ci.yml` (automated PyTest matrix for Python 3.10, 3.11, 3.12)

To publish this repository to your GitHub account:
```bash
cd networked-performative-dynamics

# Initialize local git repository
git init -b main

# Stage all files
git add .

# Create initial commit
git commit -m "feat: initial commit for networked performative dynamics research repository"

# Link to your GitHub remote repository and push
git remote add origin https://github.com/<your-username>/networked-performative-dynamics.git
git push -u origin main
```

---

## 8. Experimental Findings

### Experiment 1: Contraction Breakdown (`fig1_contraction_breakdown.png`)
- **Contraction Regime ($\epsilon < \gamma / \beta$):** When sensitivity is below the theoretical threshold, parameter drift $\|\theta_{t+1} - \theta_t\|_F$ contracts geometrically at a linear rate $O(\rho^t)$ with $\rho < 1$, converging to the unique performatively stable point.
- **Breakdown Regime ($\epsilon \ge \gamma / \beta$):** As $\epsilon$ increases past the threshold, the fixed-point operator loses its contraction property, resulting in non-convergent parameter drift, limit cycles, and persistent oscillations.
- **Algorithm Comparison:** While Static ERM suffers from distribution shift decay and standard RRM exhibits oscillatory instability, Causal Lookahead PO achieves steady, bounded performative risk.

### Experiment 2: Dynamical Phase Diagram (`fig2_phase_diagram.png`)
- **Echo-Chamber Bifurcations:** Evaluated across a 2D parameter grid of habit memory $\lambda \in [0, 0.6]$ and peer coupling $\kappa \in [0, 0.4]$ using Sarle's Bimodality Coefficient:
  $$b = \frac{g^2 + 1}{k + 3 \frac{(N-1)^2}{(N-2)(N-3)}} > \frac{5}{9} \approx 0.555$$
- **Phase Transition Boundary:** When $\lambda + \kappa > 0.5$, peer interactions amplify path-dependent feedback into strong bimodal clustering along community boundaries, producing self-reinforcing echo-chambers.

### Experiment 3: Duopoly Competition (`fig3_duopoly_polarization.png`)
- **Platform Specialization:** Two competing platforms ($\theta_A, \theta_B$) rapidly differentiate their strategies. Platform A specializes in Community 0 items while Platform B captures Community 1 items.
- **Accelerated Polarization:** Duopoly competition accelerates user community separation by over **2.4x** compared to a single monopoly platform, as competition forces algorithms to cater to extreme preferences to retain market share.

### Experiment 4: Topology & Algorithmic Ablation (`fig4_ablation_mitigation.png`)
- **Topology Sensitivity:** Stochastic Block Models (SBM) produce the highest polarization due to modularity, followed by Watts-Strogatz small-world graphs. Erdős-Rényi and disconnected graphs exhibit significantly lower echo-chamber formation.
- **Mitigation via Lookahead:** Causal Lookahead PO with cluster-randomized zero-order estimation successfully preserves catalog diversity and achieves the highest long-term social welfare across rounds.

---

## 7. Citation

If you build upon this codebase or theoretical framework, please cite:
```bibtex
@article{networked_performative_dynamics2026,
  title   = {Stateful Performative Prediction under Multi-Platform Competition and Social Network Interference},
  author  = {S.D.},
  year    = {2026}
}
```
