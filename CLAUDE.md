# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository documents critical errors found in the paper "Online Interior Point Methods for Time-Varying Equality Constraints". The main analysis is in `OIPM-TEC-review/`.

GitHub: https://github.com/LORER-MTL/online-ipm

## Repository Structure

- `OIPM-TEC-review/` - Paper review materials
  - `proof_errors_analysis.md` - Detailed analysis of proof errors
  - `main.tex` - LaTeX source for the review
  - `main.pdf` - Compiled review document

- `papers/` - Reference papers (MOSP.pdf, OIPM_JLL.pdf, OPEN-TEC-JLL.pdf)

- `online-ipm/` - Source code (for future implementation)

## Setup

```bash
# Install dependencies (uses uv package manager)
uv sync

# Alternative: traditional pip
pip install -e .
```

## Paper Errors Summary

The paper "Online Interior Point Methods for Time-Varying Equality Constraints" contains **many critical errors** in its proofs. Full details in `OIPM-TEC-review/proof_errors_analysis.md`.

### Lemma invHess (Lines 325-335) - CRITICAL

**Claims:** `‖D(y₁)D⁻¹(y₂)‖_{D(y₁)} ≤ 1/(1-‖y₁-y₂‖_{D(y₁)})²`

**Errors:**
1. **False equality (Line 328):** Claims `‖M‖_{D(y₁)} = ‖M⁻¹‖_{D(y₁)}` - a matrix and its inverse do NOT have the same operator norm. This only holds for isometries, and D(y₁)D(y₂)⁻¹ is not isometric when y₁ ≠ y₂.
2. **Dimensional inconsistency (Line 329):** LHS is a norm, RHS has squared norms - dimensionally wrong.
3. **Min vs Sup (Line 329-330):** Uses minimum instead of supremum in operator norm definition.
4. **Wrong matrix order:** Standard self-concordance bounds λ_max(D(y₁)⁻¹D(y₂)), but lemma has D(y₁)D(y₂)⁻¹ (reciprocal).

### Lemma nred (Lines 743-778) - CRITICAL

**Errors:**
1. **Unjustified inequality (Line 752):** Matrix norm application conflates operator norm with specific quadratic form ratio - unjustified and potentially incorrect.
2. **Sign error (Line 770):** Uses `y - τn` but should be `y + τn` since y⁺ = y + n.
3. **Missing factor (Line 770):** Newton step n_t(y,η) missing from integrand - fundamental theorem of calculus incorrectly applied.
4. **Wrong antiderivative (Lines 774-775):** Sign error that happens to cancel out, suggesting reverse-engineering.

### Barrier Complexity Argument (Lines 469-474) - CRITICAL

**Claims:** If `‖∇φ(x)‖²_{∇²φ(x)} ≤ v_f`, then for ANY pair (x,v): `[∇φ(x) + Aᵀv; 0]ᵀ D⁻¹ [∇φ(x) + Aᵀv; 0] ≤ v_f`

**Errors:**
1. **Wrong reference:** "Section 2.3.1" of Renegar does not contain this result.
2. **Logical gap:** Barrier complexity ≠ norm of modified gradient. Adding Aᵀv to gradient and using D⁻¹ instead of (∇²φ)⁻¹ does not preserve the bound.
3. **Cross terms:** The bound cannot hold for arbitrary v due to the term `2∇φ(x)ᵀ P Aᵀ v` which depends on v.

### Lemma yx (Lines 519-540) - CRITICAL

**Claims:** `‖y_t - y_tη‖_{D(y_t)} ≥ ‖x_t - x_tη‖_{∇²φ(x_t)}`

**Errors:**
1. **Ignoring cross terms:** D(y_t) has off-diagonal blocks A and Aᵀ. The proof assumes zeroing part of a vector makes the quadratic form smaller, but this only works for block-diagonal matrices. The cross term `2(x_t - x_tη)ᵀAᵀ(v_t - v_tη)` can be negative.
2. **Concrete counterexample:** With A=[1], ∇²φ=1, x_t-x_tη=1, v_t-v_tη=-10: LHS=-19, RHS=1, so -19 ≥ 1 is false.
3. **D(y_t) not positive definite:** Has zero block in bottom-right, making the "Hessian norm" not a proper norm.
