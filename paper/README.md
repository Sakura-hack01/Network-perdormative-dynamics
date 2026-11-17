# 2-Page Extended Abstract: Stateful Performative Prediction

This directory contains the **2-page NeurIPS/ICML-style LaTeX extended abstract** summarizing the theoretical guarantees, empirical findings, and algorithms of `networked-performative-dynamics`.

---

## 1. Directory Structure

```
paper/
├── extended_abstract.tex       # Complete 2-page LaTeX source (two-column NeurIPS/ICML workshop format)
├── references.bib              # Complete BibTeX bibliography (Perdomo, Mendler-Dünner, Ugander, Flaxman, etc.)
├── README.md                   # Compilation and Overleaf instructions
└── figures/                    # 300-DPI publication figures embedded in the abstract
    ├── fig1_contraction_breakdown.png
    ├── fig2_phase_diagram.png
    ├── fig3_duopoly_polarization.png
    └── fig4_ablation_mitigation.png
```

---

## 2. Compilation Instructions

### Option A: 1-Click Overleaf Import (Recommended)
1. Go to [Overleaf](https://www.overleaf.com/) and click **New Project -> Blank Project**.
2. Drag and drop the following files into the Overleaf file tree:
   - `extended_abstract.tex`
   - `references.bib`
   - The entire `figures/` folder (containing the `.png` files)
3. Set compiler to **pdfLaTeX** and click **Recompile**.
4. The 2-page PDF will compile with formatted figures, propositions, equations, and citations.

### Option B: Local CLI Compilation
If you have a TeX distribution (TeX Live, MacTeX, or MiKTeX) installed:
```bash
cd paper/
pdflatex extended_abstract.tex
bibtex extended_abstract
pdflatex extended_abstract.tex
pdflatex extended_abstract.tex
```

---

## 3. Abstract Outline & Key Scientific Contributions

- **Proposition 1 (Effective Sensitivity):** Proves that stateful memory $\lambda$ and social network diffusion $\tilde{A}$ scale the effective performative sensitivity to:
  $$\epsilon_{\text{eff}} \approx \frac{\epsilon}{1 - \lambda - \kappa \rho(\tilde{A})}$$
  yielding a rigorous breakdown criterion for standard Banach contraction.
- **Duopoly Competition:** Documents how competing platforms accelerate echo-chamber polarization by $>2.4\times$ through algorithmic niche capture.
- **Causal Lookahead PO:** Mitigates social network interference bias via spectral graph cut partitioning and two-point zero-order gradient estimation.
- **Empirical Benchmarks:** Complete quantitative telemetry on empirical risk, performative risk, drift, Sarle's bimodality, and social welfare across 1,000 agents.
