# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository documents critical errors found in the paper "Online Interior Point Methods for Time-Varying Equality Constraints" (OIPM-TEC) and provides numerical experiments demonstrating why naive inequality constraint extensions fail.

GitHub: https://github.com/LORER-MTL/online-ipm

## Setup and Commands

Requires Python 3.10+. Uses `uv` package manager.

```bash
uv sync                    # Install dependencies
pip install -e .           # Alternative: pip install
```

### Running Experiments

```bash
# Run all experiments (generates plots to online_ipm/results/)
uv run python -m online_ipm.experiments.run_all

# Run individual experiment
uv run python -m online_ipm.experiments.test_inequality_extensions
```

Available experiments: `test_orthonormal_basis`, `test_slack_projection`, `test_barrier_method`, `test_feasible_projection`, `test_quadratic_variation`, `test_ill_conditioned_A`, `test_inequality_extensions`

### Building LaTeX Documents

```bash
cd OIPM-TEC-review && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Code Architecture

### Problem Formulation

The codebase solves time-varying LPs of the form:
```
minimize    c^T x
subject to  Ax = b_t      (equality, time-varying RHS)
            Fx ≤ g_t      (inequality, time-varying RHS)
            x ≥ 0
```

### Core Modules (`online_ipm/`)

**`problems.py`** - Problem definitions and LP solving:
- `OnlineLPInstance`: Single time-step LP (has `n`, `p`, `m` for variables, equality, inequality counts)
- `OnlineLPProblem`: Time-varying sequence with `get_instance(t)` method
- `solve_lp()`: Exact solver via scipy.linprog (HiGHS backend)
- `create_simple_2d_problem()`, `create_medium_problem()`: Test generators

**`open_m.py`** - Linear algebra utilities for KKT systems:
- `project_onto_equality(x, A, b)`: Project onto {z : Az = b}
- `build_kkt_matrix(H, A)`, `solve_kkt_system()`: Newton step computation

### Algorithm Interface (`algorithms/base.py`)

All algorithms extend `OnlineAlgorithm` with:
- `initialize(instance, x_init)`: Setup initial state
- `step(instance, x_star, f_star)`: One iteration, returns `StepMetrics`
- `get_current_x()`: Return current solution

`StepMetrics` tracks: regret, cumulative_regret, distance_to_optimum, equality_violation, inequality_violation, plus algorithm-specific metrics in `extra` dict.

### Algorithm Implementations

**Failure mode demonstrations** (show why naive extensions fail):
- `slack_projection.py` - Clipping slack variables destroys equality constraints
- `barrier_method.py` - Full Newton step exits feasible region

**Working approaches**:
- `feasible_projection.py` - QP projection preserves all constraints
- `primal_dual_line_search.py` - Standard IPM with backtracking line search
- `infeasible_start_clipping.py` - Infeasible-start Newton with residual tracking

## Repository Structure

- `OIPM-TEC-review/` - Paper review with LaTeX source and proof error analysis
- `OPEN-M-review/` - OPEN-M paper correctness analysis
- `OCO-Routing-review/` - OCO-Routing paper analysis (depends on OIPM-TEC)
- `papers/` - Reference papers (MOSP.pdf, OIPM_JLL.pdf, OPEN-TEC-JLL.pdf, OCO_Routing.pdf)
- `online_ipm/` - Python package (algorithms, experiments, results)

## Paper Analysis Summary

| Aspect | OPEN-M | OIPM-TEC | OCO-Routing |
|--------|--------|----------|-------------|
| Proofs correct? | Yes (given orthonormal F_t) | Contains errors | Relies on OIPM-TEC (invalid) |
| Time-varying A_t? | YES | NO (constant A) | YES (via b_t) |
| Practical applicability | Narrow | N/A (proofs invalid) | May work as heuristic |

**OIPM-TEC**: Contains critical errors in Lemmas invHess, nred, yx, and barrier complexity argument. See `OIPM-TEC-review/proof_errors_analysis.md`.

**OPEN-M**: Mathematically sound but orthonormality of F_t (null space basis) is a required assumption—not "without loss of generality." Bounds inflate by κ(F_t)³ otherwise. See `OPEN-M-review/open_m_correctness_analysis.md`.

**OCO-Routing**: Theoretical guarantees invalid (relies on OIPM-TEC), but may work in practice as a heuristic. See `OCO-Routing-review/oco_routing_analysis.md`.
