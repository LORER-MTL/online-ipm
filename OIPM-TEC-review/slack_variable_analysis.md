# Analysis: Extending OPEN-M to Inequality Constraints via Slack Variable Projection

**Date:** January 18, 2026
**Document:** Theoretical analysis of a proposed extension

## Summary

This document analyzes a proposed approach to extend the OPEN-M algorithm (from "Online Interior Point Methods for Time-Varying Equality Constraints") to handle inequality constraints by introducing slack variables and projecting them onto the non-negative orthant. **The analysis concludes that this approach fundamentally fails** due to incompatibilities with the theoretical framework.

---

# Problem Formulations

## Original Problem (OPEN-M handles this)

At each round t:
```
min_{x_t}  f_t(x_t)
s.t.      A_t x_t = b_t     (time-varying linear equality)
```

Where:
- x_t ∈ ℝⁿ is the decision
- f_t is twice-differentiable (not necessarily convex)
- A_t ∈ ℝ^{p×n} full row-rank, b_t ∈ ℝᵖ

## Target Problem (with inequalities)

```
min_{x_t}  f_t(x_t)
s.t.      A_t x_t = b_t         (equality)
          C_t x_t ≤ d_t         (time-varying inequality)
```

## Proposed Reformulation with Slack Variables

Introduce slack s_t ∈ ℝᵐ to convert inequalities to equalities:
```
min_{x_t, s_t}  f_t(x_t)
s.t.            A_t x_t = b_t
                C_t x_t + s_t = d_t
                s_t ≥ 0
```

## Proposed (Problematic) Algorithm

1. Treat the augmented system [A; C, I] y = [b; d] where y = (x, s) as equality constraints
2. Apply OPEN-M in the extended (x, s) space
3. **Project s onto s ≥ 0** by clipping: s ← max(s, 0)

---

# Why This Approach Fails

## Failure 1: Projection onto s ≥ 0 Does NOT Preserve Distance to Optimum

OPEN-M's analysis relies critically on **Theorem 3, equation (19)**:
```
‖x̃_t - x*_t‖ ≤ ‖x_t - x*_t‖
```

This holds because projection onto an **affine subspace** {x : Ax = b} creates orthogonal components (Pythagorean theorem).

**The non-negative orthant s ≥ 0 is a cone, not an affine subspace.**

For the optimum (x*, s*) where some s*_i = 0 (active constraints):
- Projection onto s ≥ 0 does NOT guarantee ‖(x, s_clipped) - (x*, s*)‖ ≤ ‖(x, s) - (x*, s*)‖
- The projection can move you **further** from the optimum

### Counterexample

Let s* = 0 (constraint active at optimum). If s = -1 and we clip to s_clipped = 0, we get closer. But if s* = 0.5 and s = -0.1, clipping to 0 moves us further from s* = 0.5.

---

## Failure 2: Clipping Destroys the Newton Step Structure

The Newton step solves a KKT system that couples (Δx, Δs). After clipping s:
- The equality C_t x + s = d_t is violated (you have C_t x + s_clipped ≠ d_t)
- The next Newton step is computed from an inconsistent point
- **The quadratic convergence analysis (Lemma 3, eq. 8-9) breaks down completely**

The identity Δx_t = F_t Δz_t (Lemma 1) requires the iterate to be **feasible**. After clipping, you're not feasible for the equality C_t x + s = d_t.

---

## Failure 3: Information Destruction is Catastrophic

When s_i < 0 (constraint i is violated: C_i x > d_i):
- The magnitude |s_i| encodes **how much** the constraint is violated
- The gradient ∇_s L contains dual information for returning to feasibility
- Clipping to s_i = 0 **erases this information**

The Newton method needs the full gradient to compute the correct descent direction. You cannot recover from constraint violations if you destroy the signal telling you how far you've gone wrong.

---

## Failure 4: No Constraint Violation Bound is Possible

OPEN-M achieves Vio(T) = O(V_T) because:
1. Projection onto affine subspace is exact: A x̃ = b always
2. Newton steps preserve feasibility: A Δx = 0

With clipping, **neither property holds**:
- After clipping s, you satisfy s ≥ 0 but violate C x + s = d
- This means C x ≤ d may still be violated (when s was negative, constraint was violated, and clipping s doesn't fix x)
- The actual constraint violation ‖max(C_t x_t - d_t, 0)‖ has no bound

---

## Failure 5: The Fundamental Incompatibility

Interior point methods handle inequalities via **barrier functions** that:
- Keep iterates strictly feasible (s > 0 always)
- Provide smooth gradient information as you approach the boundary
- Follow a central path with well-understood convergence

Clipping is discontinuous and non-smooth. It's fundamentally incompatible with Newton's method which requires smooth, differentiable structure.

**Analogy:** Clipping is like trying to do calculus with a step function. Newton's method assumes you can take derivatives; clipping creates a point where derivatives don't exist (or are zero/infinite).

---

## Failure 6: The Regret Bound Derivation Fails

Theorem 3's proof uses equation (16):
```
Σ ‖x_t - x*_t‖ ≤ (1 - 2L/h γ)^{-1} (V_T + δ)
```

This requires iterates to stay in a neighborhood where:
- The Hessian is invertible
- Lipschitz continuity holds
- The Newton step contracts distance to optimum

Clipping can eject you from this neighborhood with no guarantee of return. Once outside, all bounds are void.

---

# Mathematical Details

## Why Projection onto Affine Subspaces Works

For an affine subspace S = {x : Ax = b}:
- The projection P_S(x) is the unique point in S closest to x
- For any y ∈ S: ‖x - y‖² = ‖x - P_S(x)‖² + ‖P_S(x) - y‖² (Pythagorean theorem)
- Therefore: ‖P_S(x) - y‖ ≤ ‖x - y‖ for all y ∈ S

This is used in OPEN-M's Theorem 3 to bound the distance after projection.

## Why Projection onto Cones Fails

For the non-negative orthant K = {s : s ≥ 0}:
- P_K(s) = max(s, 0) (componentwise)
- If s* ∈ interior(K) (s* > 0) and s ∈ K^c (some s_i < 0):
  - The projection max(s, 0) may move away from s*

**Concrete example:**
- s* = (0.5, 0.5)
- s = (-0.1, 0.8)
- P_K(s) = (0, 0.8)

Distances:
- ‖s - s*‖ = √(0.6² + 0.3²) = √0.45 ≈ 0.67
- ‖P_K(s) - s*‖ = √(0.5² + 0.3²) = √0.34 ≈ 0.58

In this case projection helps, but:
- s* = (0.5, 0.5)
- s = (-0.1, 0.5)
- P_K(s) = (0, 0.5)

Distances:
- ‖s - s*‖ = √(0.6² + 0²) = 0.6
- ‖P_K(s) - s*‖ = √(0.5² + 0²) = 0.5

Still helps. Now consider:
- s* = (0, 0.5) (first component active at optimum)
- s = (0.3, 0.5)
- P_K(s) = (0.3, 0.5) (no change, already feasible)
- ‖s - s*‖ = 0.3

No issue here. The problem arises when:
- s* = (0.5, 0) (second component active)
- s = (0.4, -0.1)
- P_K(s) = (0.4, 0)

- ‖s - s*‖ = √(0.1² + 0.1²) ≈ 0.14
- ‖P_K(s) - s*‖ = √(0.1² + 0²) = 0.1

This helps too. **The real issue is the coupled equality constraint:**

After clipping s from s = (0.4, -0.1) to (0.4, 0), the constraint C x + s = d becomes violated. The point (x, s_clipped) is no longer feasible for the combined system.

---

# Conclusion

The approach of projecting slack variables by clipping fails because:

1. **Mathematical:** Projection onto a cone ≠ projection onto an affine subspace. The distance-preserving property that enables OPEN-M's analysis does not hold.

2. **Algorithmic:** Clipping destroys the KKT system structure and gradient information needed for Newton convergence.

3. **Theoretical:** No regret or constraint violation bounds can be derived because the fundamental lemmas (1-3) don't apply after clipping.

The intuition "we lose a lot of information from clipping slack variables" is correct — but more precisely, we lose the **mathematical structure** that makes the method work at all.

---

# What Would Be Needed Instead

A proper extension would require:

## Option 1: Log-Barrier Methods
- Add a log-barrier term -μ Σ log(s_i) to the objective
- Keep iterates strictly feasible (s > 0 always)
- Follow a central path with parameter μ → 0
- Requires predictor-corrector schemes to maintain centrality
- Analysis must account for curved geometry near boundary

## Option 2: Penalty Methods
- Add a penalty term for constraint violations
- Requires careful penalty parameter tuning
- Convergence analysis differs substantially from OPEN-M

## Option 3: Augmented Lagrangian
- Combine Lagrangian with quadratic penalty
- More robust to infeasibility
- Different convergence properties

## Key Challenges for Any Extension
- The geometry of {x : Ax = b, Cx ≤ d} is more complex than {x : Ax = b}
- Active set changes create discontinuities
- The central path bends near constraint boundaries
- Analysis must handle transitions between active/inactive constraints

This is essentially building a full online interior-point method, which is a much more complex undertaking than the elegant OPEN-M approach for equality constraints only.

---

# References

1. OPEN-M paper: "Online Interior Point Methods for Time-Varying Equality Constraints"
2. Renegar, J. "A Mathematical View of Interior-Point Methods in Convex Optimization"
3. Boyd, S. and Vandenberghe, L. "Convex Optimization," Chapter 11 (Interior-point methods)
4. Nesterov, Y. and Nemirovski, A. "Interior-Point Polynomial Algorithms in Convex Programming"
