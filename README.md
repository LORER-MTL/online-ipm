# Online Interior Point Method

Critical analysis of the paper "Online Interior Point Methods for Time-Varying Equality Constraints" (OIPM-TEC).

GitHub: https://github.com/LORER-MTL/online-ipm

## Repository Structure

- `OIPM-TEC-review/` - Paper review and analysis
  - `main.tex` / `main.pdf` - LaTeX review document
  - `proof_errors_analysis.md` - Detailed analysis of proof errors
  - `slack_variable_analysis.md` - Why slack variable projection fails for inequalities
  - `barrier_reformulation_analysis.md` - Why barrier reformulation fails for inequalities
  - `open_m_correctness_analysis.md` - Analysis of OPEN-M paper correctness
- `papers/` - Reference papers (MOSP.pdf, OIPM_JLL.pdf, OPEN-TEC-JLL.pdf)
- `online_ipm/` - Source code and numerical experiments

## Setup

```bash
# Install dependencies (uses uv package manager)
uv sync

# Alternative: traditional pip
pip install -e .
```

## Running Experiments

```bash
# Run all numerical experiments
uv run python -m online_ipm.experiments.run_all

# Run individual experiments
uv run python -m online_ipm.experiments.test_orthonormal_basis
uv run python -m online_ipm.experiments.test_slack_projection
uv run python -m online_ipm.experiments.test_barrier_method
```

## Building the LaTeX Document

```bash
cd OIPM-TEC-review
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Key Findings

### OIPM-TEC Paper: Contains Critical Errors

The OIPM-TEC paper contains **critical errors** in its proofs (Lemmas invHess, nred, yx, and barrier complexity argument). See `OIPM-TEC-review/proof_errors_analysis.md` for details.

### OPEN-M Paper: Correct but Misleading

The original OPEN-M paper (time-varying equality constraints) is mathematically sound but contains a misleading claim: "Without loss of generality, we let F_t = F̄_t" (orthonormal basis). Orthonormality is a **required assumption**, not optional—bounds inflate by κ(F_t)³ otherwise.

See `OIPM-TEC-review/open_m_correctness_analysis.md` for details.
