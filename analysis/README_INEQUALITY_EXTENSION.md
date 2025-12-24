# OIPM Extension to Inequality Constraints

## Quick Summary

This analysis shows how to extend the **Online Interior Point Method (OIPM)** to handle **inequality constraints** using slack variables, and proves that the performance guarantees from Theorem 1 transfer to the original problem.

## Main Result

**Original Problem:**
```
minimize    c^T x
subject to  Ax = b_t    (equalities)
            Fx ≤ g_t    (inequalities)
```

**Performance Guarantee:**
```
Dynamic Regret:  R_d(T) ≤ O(p) + ||c|| · [(1 + ||F||)·V_T^x + V_g]

Constraint Violation:  Vio(T) ≤ V_b^eq + V_g^ineq
```

Where:
- `p` = number of inequality constraints
- `V_T^x` = path variation of optimal solutions
- `V_g` = variation in inequality RHS
- `||F||` = norm of inequality constraint matrix

## Files in This Analysis

### 1. Mathematical Derivation
**File:** `inequality_extension.md`

Complete mathematical proof showing:
- Problem transformation via slack variables
- Application of Theorem 1 to augmented problem
- Transfer of guarantees back to original problem
- Detailed bound analysis with all constants

**Key sections:**
- §1: Problem setup (original vs augmented)
- §2: Barrier method formulation
- §3: Applying Theorem 1
- §4: Slack variable coupling analysis
- §5: Performance guarantees for original problem

### 2. Numerical Demonstration
**File:** `inequality_extension_demo.py`

Python implementation that:
- Transforms LPs to slack form
- Computes theoretical bounds
- Verifies slack coupling inequality
- Checks original problem feasibility
- Generates visualization

**Key functions:**
- `transform_to_slack_form()` - Converts LP with inequalities
- `compute_theoretical_regret_bound()` - Evaluates Theorem 1 bounds
- `analyze_slack_coupling()` - Verifies V_s ≤ V_g + ||F||·V_x
- `verify_original_feasibility()` - Checks constraint satisfaction

**Usage:**
```bash
python analysis/inequality_extension_demo.py
```

### 3. Visualization
**File:** `inequality_extension_demo.png`

4-panel figure showing:
1. Regret bound decomposition (constant vs path-dependent)
2. Path variation components (V_x, V_g, V_s)
3. Slack coupling bound verification
4. Problem parameters table

### 4. Executive Summary
**File:** `INEQUALITY_EXTENSION_SUMMARY.md`

High-level overview with:
- Main results
- Numerical verification
- Practical implications
- When to use this approach

## Key Insights

### 1. Slack Transformation is Natural

The augmented problem:
```
minimize    [c; 0]^T [x; s]
subject to  [A  0] [x]   [b_t]
            [F  I] [s] = [g_t]
            s > 0
```

Is **exactly the form** considered by the paper (equality constraints + simple bounds).

### 2. Guarantees Transfer with Coupling Factor

The slack variables couple to original variables:
```
s_t = g_t - Fx_t
```

This introduces factor `(1 + ||F||)` in the regret bound, which is tight.

### 3. Feasibility is Maintained

The barrier method ensures:
- `Ax_t = b_t` exactly (equality constraints)
- `Fx_t = g_t - s_t < g_t` exactly (inequality constraints)
- `s_t > 0` (slack positivity)

So the original problem's constraints are **always satisfied**.

### 4. Regret Depends on Problem Dynamics

The bound grows with:
- Path variation `V_T^x` (how much optimal solutions change)
- Constraint variation `V_g` (how much inequality RHS changes)
- Matrix norm `||F||` (coupling strength)

## Numerical Example

Running the demo with `n=5, m=2, p=3, T=20`:

```
Slack variable coupling analysis:
  ||F|| = 2.6747
  V_x (path variation of x) = 1.0636
  V_s (path variation of slacks) = 3.1310
  V_s upper bound = 5.5775
  Bound slack = 2.4465  ✓ (bound holds with room to spare)

Theoretical regret bound (β=1.1, η₀=1.0):
  Constant term: 72.6000
  Path-dependent term: 11.6168
  Total bound: 84.2168

Original problem feasibility check:
  Equality constraints satisfied: True
    (residual: 1.34e-15)
  Inequality constraints satisfied: True
    (min slack: 0.5562)
```

## Theoretical Contributions

This analysis demonstrates:

1. **Generality of OIPM:** The framework handles any LP, not just equality-constrained ones

2. **Clean theory:** Slack transformation preserves self-concordance and allows direct application of Theorem 1

3. **Tight bounds:** The coupling factor `(1 + ||F||)` is necessary and captures the true cost of tracking inequalities

4. **Feasibility guarantee:** Unlike some online algorithms, OIPM maintains exact feasibility at all times

## Practical Recommendations

**Use slack transformation when:**
- ✓ Need theoretical guarantees on regret and feasibility
- ✓ Inequality RHS varies over time: `g_t`
- ✓ Problem size allows dimension increase `n → n+p`
- ✓ Matrix norm `||F||` is not too large

**Consider alternatives when:**
- ✗ Problem is very large (`p` is huge)
- ✗ `||F||` is very large (coupling factor degrades bound)
- ✗ Only care about approximate feasibility
- ✗ Inequalities are fixed (can use simpler barrier)

## Reproducibility

All results are fully reproducible:

```bash
# 1. Run numerical demonstration
cd /home/willyzz/Documents/online-ipm
python analysis/inequality_extension_demo.py

# 2. Output includes:
#    - Numerical verification of all bounds
#    - Feasibility checks
#    - Visualization (saved to .png)
```

**Dependencies:**
- numpy (linear algebra)
- matplotlib (visualization)

## Future Extensions

Possible directions:
1. **Time-varying A and F matrices** (beyond just b_t and g_t)
2. **Nonlinear inequalities** with general self-concordant barriers
3. **Stochastic constraints** with probabilistic guarantees
4. **Warm-starting** to reduce constant term in regret bound

## References

- Original paper: "Online Interior Point Methods for Time-Varying Equality Constraints"
- Self-concordant functions: Nesterov & Nemirovskii (1994)
- Interior point methods: Boyd & Vandenberghe (2004)

## Contact

For questions or discussions about this analysis, please refer to the original repository.

---

**Last updated:** December 8, 2025
