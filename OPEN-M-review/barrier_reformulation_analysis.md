# Analysis: Applying OPEN-M to Barrier Reformulation (Non-Iterative)

## OPEN-M Algorithm Summary

OPEN-M solves online optimization with time-varying linear **equality** constraints:
```
min_x  f_t(x)
s.t.   A_t x = b_t
```

**Algorithm (each round t):**
1. **Play** decision x_t
2. **Observe** f_t(x_t) and constraints A_t, b_t
3. **Project** onto new feasible set: x̃_t = x_t + A_t^T (A_t A_t^T)^{-1} (b_t - A_t x_t)
4. **Newton update**: [x_{t+1}; ν_t] = [x̃_t; 0] - D_t(x̃_t)^{-1} [∇f_t(x̃_t); 0]

Where D_t(x) = [∇²f_t(x), A_t^T; A_t, 0] is the KKT matrix.

**Key properties enabling the analysis:**
- Projection onto affine subspace preserves distance: ‖x̃_t - x*_t‖ ≤ ‖x_t - x*_t‖ (Pythagorean theorem)
- Newton step contracts: ‖x_{t+1} - x*_t‖ ≤ (2L/h) ‖x_t - x*_t‖²
- Achieves R_d(T) = O(V_T + 1) and Vio(T) = O(V_T + 1)

**Required assumptions:**
1. ‖∇²f(x*)^{-1}‖ ≤ 1/h (bounded inverse Hessian at optimum)
2. ‖∇²f(x) - ∇²f(x*)‖ ≤ L‖x - x*‖ for ‖x - x*‖ ≤ β (Lipschitz Hessian)
3. ‖f(x) - f(x*)‖ ≤ l‖x - x*‖ (Lipschitz function values)

---

## The Proposed Approach

**Target problem (with inequalities):**
```
min_x  f(x)
s.t.   Ax = b
       Cx ≤ d
```

**Barrier reformulation with fixed large μ:**
```
min_x  f(x) + (1/μ) Σ_i -log(d_i - C_i x)
s.t.   Ax = b
```

The idea: For large μ, the barrier term is negligible in the interior, so the barrier optimum approximates the constrained optimum. Apply OPEN-M to this reformulated problem (treating it as equality-constrained with a modified objective).

**Key difference from OIPM:** No iterative barrier parameter schedule. Use a single, large μ throughout.

---

## Why This Approach Fails: Complete List of Issues

### Issue 1: Newton Step Can Exit the Feasible Region — CRITICAL

In standard optimization, you **never** take a full Newton step with barrier methods. You always do a **line search** to ensure:
- The new point satisfies Cx < d (strict inequality)
- The barrier function remains defined (no log of negative numbers)

**OPEN-M takes the full Newton step without line search.**

If the Newton step Δx is such that C(x + Δx) ≥ d for any constraint i:
- The barrier function becomes undefined (+∞ or complex)
- The algorithm crashes or produces meaningless results
- There's no recovery mechanism

**Intuition from normal optimization:** Even offline, barrier Newton methods require damped steps near the boundary. The "pure" Newton step regularly violates constraints.

---

### Issue 2: Hessian Conditioning Catastrophe — CRITICAL

The barrier Hessian is:
```
∇²φ_barrier(x) = (1/μ) Σ_i C_i^T C_i / (d_i - C_i x)²
```

**For large μ:**
- **In the interior** (far from boundary): Barrier Hessian ≈ 0, total Hessian ≈ ∇²f(x)
- **Near the boundary**: Barrier Hessian dominates but is O(1/(d-Cx)²) — extremely large

This creates a **condition number catastrophe:**
- The total Hessian varies from ≈ ∇²f(x) to ≈ ∞ depending on position
- Condition number: κ = O(μ · (d - Cx)^{-2}) which is unbounded
- Newton steps become numerically unstable

**In standard IPM:** This is handled by:
1. Starting with small μ (well-conditioned)
2. Gradually increasing μ while staying on the central path
3. The central path provides "warm starts" for each new μ

With fixed large μ from the start, you get the worst conditioning with no warm start.

---

### Issue 3: Assumption 2 (Lipschitz Hessian) Fails — CRITICAL

OPEN-M requires: ‖∇²f(x) - ∇²f(x*)‖ ≤ L‖x - x*‖ for ‖x - x*‖ ≤ β

For the barrier function:
```
∂/∂x [1/(d_i - C_i x)²] = 2 C_i / (d_i - C_i x)³
```

**Near the boundary, this derivative blows up as 1/(d-Cx)³.**

No finite Lipschitz constant L exists for the barrier Hessian near the boundary. The assumption fundamentally fails.

**Consequence:** Lemma 3's Newton contraction bound (eq. 8-9) doesn't hold. The core convergence analysis breaks down.

---

### Issue 4: The Convergence Neighborhood Shrinks to Zero — CRITICAL

OPEN-M's Theorem 3 requires iterates to stay in:
```
‖x_t - x*_t‖ ≤ γ = min{β, h/(2L)}
```

For the barrier problem with large μ:
- β (Lipschitz ball radius) → 0 near boundary
- L (Lipschitz constant) → ∞ near boundary
- Therefore γ → 0

**The "basin of attraction" for Newton convergence vanishes.** Even if you start very close to the barrier optimum, slight perturbations (from time-varying objectives) can eject you from this neighborhood.

---

### Issue 5: Barrier Optimum ≠ Constrained Optimum — FUNDAMENTAL

Even with arbitrarily large μ, the barrier optimum x*(μ) satisfies Cx*(μ) < d (strict inequality).

If the true constrained optimum has active constraints (Cx* = d for some i), then:
```
‖x*(μ) - x*‖ > 0  for all finite μ
```

**The gap decreases as O(1/μ) but never vanishes.**

For the online setting:
- You're tracking a sequence of barrier optima x*_t(μ), not true optima x*_t
- The regret is measured against true optima
- There's an irreducible O(T/μ) gap in regret from this approximation alone

---

### Issue 6: No Projection Step Preserves Barrier Structure

OPEN-M's projection onto {x : Ax = b} works because:
1. It's a closed-form linear operation
2. The optimum lies in this subspace
3. Projection reduces distance to optimum (Pythagorean theorem)

**For the barrier problem:**
- The optimum lies in {x : Ax = b, Cx < d} (strict interior w.r.t. inequalities)
- If the Newton step exits Cx < d, there's no "projection" back that:
  - Preserves the barrier structure
  - Reduces distance to optimum
  - Has a closed form

You could project onto {x : Ax = b}, but this doesn't ensure Cx < d.

---

### Issue 7: Optimum Variation V_T Changes Meaning

OPEN-M bounds regret in terms of V_T = Σ‖x*_{t+1} - x*_t‖.

For the barrier problem:
- V_T(μ) = Σ‖x*_{t+1}(μ) - x*_t(μ)‖ (barrier optima variation)
- V_T = Σ‖x*_{t+1} - x*_t‖ (true optima variation)

**These are different quantities!**

- x*(μ) depends nonlinearly on μ and constraint geometry
- Even if true optima move slowly, barrier optima can move faster (or slower)
- The relationship between V_T(μ) and V_T is unclear

---

### Issue 8: The "Large μ Limit" is Singular

**From normal optimization intuition:**

The limit μ → ∞ is a **singular perturbation**:
- For any finite μ, the barrier problem is smooth and well-posed
- At μ = ∞, the problem becomes the original constrained problem (non-smooth at boundary)
- The transition is discontinuous in important ways

You cannot simply "take μ large enough" and expect good behavior. The problem structure changes qualitatively:
- For finite μ: smooth, unconstrained (on the equality subspace), interior optimum
- At μ = ∞: non-smooth, constrained, boundary optimum possible

**Standard IPM navigates this via the central path.** Direct application with large μ skips this navigation.

---

### Issue 9: Assumption 1 (Bounded Inverse Hessian) Becomes Problem-Dependent

The barrier Hessian at x*(μ) is:
```
H(x*(μ)) = ∇²f(x*(μ)) + (1/μ) Σ_i C_i^T C_i / (d_i - C_i x*(μ))²
```

For large μ, x*(μ) is close to the boundary for active constraints. Let s_i = d_i - C_i x*(μ) be the slack. Then:
- s_i = O(1/μ) for active constraints
- The barrier Hessian contribution is O(1/μ · μ²) = O(μ) — grows with μ!

**The Hessian doesn't converge as μ → ∞.** It either:
- Stays O(1) if no constraints are active (but then why use barrier?)
- Blows up as O(μ) if constraints are active

This means Assumption 1's constant h depends on μ, and h → 0 as μ → ∞.

---

### Issue 10: No Natural Initialization

OPEN-M requires: ‖x_0 - x*_0‖ ≤ γ

For the barrier problem with large μ:
- x*_0(μ) is very close to the constraint boundary (for active constraints)
- Finding such an x_0 is essentially solving the barrier problem once
- The initialization problem is as hard as the original problem

**In standard IPM:** You start with small μ where the barrier optimum is well-interior, easy to find or approximate.

---

## Summary Table

| Issue | Severity | Standard IPM Fix | Available in OPEN-M? |
|-------|----------|------------------|---------------------|
| Newton step exits feasible region | CRITICAL | Line search | No |
| Hessian conditioning | CRITICAL | Central path, gradual μ increase | No |
| Lipschitz Hessian fails | CRITICAL | Stay on central path | No |
| Convergence neighborhood → 0 | CRITICAL | Predictor-corrector | No |
| Barrier ≠ constrained optimum | FUNDAMENTAL | Accept O(1/μ) error | Maybe (cost: O(T/μ) regret) |
| No projection for inequalities | STRUCTURAL | Interior-point structure | No |
| V_T meaning changes | ANALYTICAL | N/A | No |
| Singular limit | FUNDAMENTAL | Central path | No |
| Hessian bound diverges | CRITICAL | N/A | No |
| Initialization difficulty | PRACTICAL | Start with small μ | No |

---

## Conclusion

**The approach fundamentally fails** because OPEN-M's analysis relies on:

1. **Full Newton steps** — barrier methods require line search
2. **Lipschitz Hessian** — barrier Hessian is not Lipschitz near boundary
3. **Well-conditioned Hessian** — barrier Hessian has catastrophic conditioning for large μ
4. **Projection preserves distance** — no such projection exists for interior of inequalities

**The core insight:** Interior-point methods work by *carefully navigating* the central path. OPEN-M tries to *skip* this navigation by going directly to large μ, but the mathematical structure required for Newton convergence doesn't exist at large μ without the central path's guidance.

**Normal optimization intuition:** You never start IPM with large μ. You always follow the central path. There's a reason for this — it's not just computational convenience, it's mathematical necessity.
