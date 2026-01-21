# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository documents critical errors found in the paper "Online Interior Point Methods for Time-Varying Equality Constraints" (OIPM-TEC). The main analysis and review materials are in `OIPM-TEC-review/`.

GitHub: https://github.com/LORER-MTL/online-ipm

## Repository Structure

- `OIPM-TEC-review/` - Paper review, analysis documents, and LaTeX source
- `papers/` - Reference papers (MOSP.pdf, OIPM_JLL.pdf, OPEN-TEC-JLL.pdf)
- `online_ipm/` - Source code and numerical experiments

## Setup and Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Alternative: traditional pip
pip install -e .
```

### Running Experiments

```bash
# Run all numerical experiments (inequality extension failures)
uv run python -m online_ipm.experiments.run_all

# Run individual experiments
uv run python -m online_ipm.experiments.test_orthonormal_basis
uv run python -m online_ipm.experiments.test_slack_projection
uv run python -m online_ipm.experiments.test_barrier_method
uv run python -m online_ipm.experiments.test_feasible_projection
uv run python -m online_ipm.experiments.test_quadratic_variation
```

### Building the LaTeX Document

```bash
cd OIPM-TEC-review
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Code Architecture

The `online_ipm/` package implements numerical experiments demonstrating why naive inequality extensions fail:

**Core modules:**
- `problems.py` - LP instance definitions (`OnlineLPInstance`, `OnlineLPProblem`) and test problem generators
- `open_m.py` - Shared utilities: `project_onto_equality()`, `build_kkt_matrix()`, `solve_kkt_system()`

**Algorithm implementations** (all extend `OnlineAlgorithm` base class in `algorithms/base.py`):
- `algorithms/slack_projection.py` - Demonstrates failure of slack variable approach (clipping destroys equality constraints)
- `algorithms/barrier_method.py` - Demonstrates failure of barrier reformulation (full Newton exits feasible region)
- `algorithms/feasible_projection.py` - Feasible space projection algorithm

**Experiments** in `experiments/`:
- `run_all.py` - Runs all experiments and generates plots to `online_ipm/results/`
- Individual test files validate specific failure modes

**Key pattern:** All algorithms implement `initialize()`, `step()`, and `get_current_x()`. The `step()` method returns `StepMetrics` with regret, constraint violations, and algorithm-specific metrics in `extra`.

## Paper Errors Summary

The paper contains **critical errors** in its proofs. Full details in `OIPM-TEC-review/proof_errors_analysis.md`.

### Lemma invHess (Lines 325-335)
Claims `‖D(y₁)D⁻¹(y₂)‖_{D(y₁)} ≤ 1/(1-‖y₁-y₂‖_{D(y₁)})²`

**Critical errors:**
1. False equality `‖M‖_{D(y₁)} = ‖M⁻¹‖_{D(y₁)}` — only holds for isometries
2. Dimensional inconsistency (norm vs squared norm)
3. Uses minimum instead of supremum in operator norm definition
4. Wrong matrix order vs standard self-concordance bounds

### Lemma nred (Lines 743-778)
**Critical errors:**
1. Unjustified matrix norm inequality (Line 752)
2. Sign error: uses `y - τn` instead of `y + τn` (Line 770)
3. Missing Newton step n_t(y,η) in integrand
4. Wrong antiderivative sign (happens to cancel out)

### Barrier Complexity Argument (Lines 469-474)
Claims bounds on `[∇φ(x) + Aᵀv; 0]ᵀ D⁻¹ [∇φ(x) + Aᵀv; 0]` for any (x,v).

**Critical errors:**
1. Wrong/nonexistent reference ("Section 2.3.1" of Renegar)
2. Logical gap: barrier complexity ≠ modified gradient norm
3. Cross term `2∇φ(x)ᵀ P Aᵀ v` makes bound fail for arbitrary v

### Lemma yx (Lines 519-540)
Claims `‖y_t - y_tη‖_{D(y_t)} ≥ ‖x_t - x_tη‖_{∇²φ(x_t)}`

**Critical errors:**
1. Ignores cross terms: D(y_t) has off-diagonal blocks A and Aᵀ
2. Counterexample: A=[1], ∇²φ=1, x_t-x_tη=1, v_t-v_tη=-10 gives LHS=-19, RHS=1
3. D(y_t) is not positive definite (zero block in bottom-right)

## OPEN-M Analysis Summary

The original OPEN-M paper (time-varying equality constraints) has been analyzed for correctness. Full details in `OIPM-TEC-review/open_m_correctness_analysis.md`.

**Key Finding:** OPEN-M claims to handle time-varying A_t and its proofs appear **mathematically sound**, but contain a **misleading claim**.

### The "Without Loss of Generality" Issue

The paper claims: "Without loss of generality, we let F_t = F̄_t" (orthonormal basis of null(A_t)).

**This is misleading** — orthonormality is a **required assumption**, not optional:
- Lemma 2's bounds depend on σ_min(F_t) = ‖F_t‖ = 1
- With non-orthonormal F_t, bounds inflate by κ(F_t)³
- Regret becomes O(κ(F_t)·V_T + 1) instead of O(V_T + 1)

**Numerical verification:** Run `uv run python -m online_ipm.experiments.test_orthonormal_basis`

**Note:** The orthonormal basis F_t is for the **null space of the constraint matrix A_t** (used in reduced-space Newton), NOT the objective function. This is an implementation requirement, not a problem-class restriction.

### Practical Limitations That Make OPEN-M Less Impressive

Beyond the orthonormal basis issue, OPEN-M has restrictive assumptions:

| Limitation | Impact |
|------------|--------|
| Uniform bounds (h, L, l) | Must hold for ALL objectives — adversarial f_t can break this |
| Variation bound v ≤ γ - (2L/h)γ² | Can be essentially zero — only nearly-static problems |
| Constraint violation O(V_T) | Constraints are violated, not satisfied |
| Near-optimal initialization | Requires ‖x_0 - x*_0‖ ≤ γ — chicken-and-egg problem |
| Single Newton step | No recovery if knocked out of γ-neighborhood |

**Bottom line:** The proofs are correct, but the O(V_T + 1) regret bound applies only to slowly-varying, well-conditioned problems where you start near-optimal and tolerate constraint violations.

### Comparison: OPEN-M vs OIPM-TEC

| Aspect | OPEN-M | OIPM-TEC |
|--------|--------|----------|
| Time-varying A_t? | YES | NO (constant A) |
| Proofs correct? | Yes (given orthonormal F_t) | Contains errors |
| Orthonormal basis? | Required (hidden) | N/A |
| Practical applicability? | Narrow (see limitations above) | N/A (proofs invalid) |
