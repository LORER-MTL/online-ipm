# Analysis: Extending OPEN-M via Feasible Space Projection

**Date:** January 20, 2026
**Document:** Analysis of projecting onto the full feasible set instead of clipping slack variables

---

## Executive Summary

This document analyzes whether **projecting onto the full feasible space** F = {(x,s) : Ax = b, Cx + s = d, s ≥ 0} can fix the problems with the clipping approach for extending OPEN-M to inequality constraints.

**Key finding:** Full projection onto F satisfies the distance-preservation property that OPEN-M requires, BUT there are several other critical issues that make this approach fail in practice.

**Important context:** The OPEN-M paper itself contains numerous errors (see Section 5). Even if the projection approach worked perfectly, the underlying algorithm has fundamental limitations.

---

## 1. The Problem: Clipping vs Full Projection

### 1.1 The Clipping Approach (Current Implementation)

The existing `SlackProjectionAlgorithm` does:

```python
# After Newton step
s_new = y_new[n:]          # Slack after Newton step
s_clipped = np.maximum(s_new, 0)  # Clip to non-negative

# This violates: Cx + s = d
# Because s was computed to satisfy Cx + s_new = d
# But s_clipped ≠ s_new (when s_new < 0)
```

**Why clipping fails:**
1. Violates the equality constraint Cx + s = d
2. Can increase distance to optimum
3. Destroys gradient information about constraint violations

### 1.2 The Proposed Full Projection Approach

Instead of clipping, project onto the **complete feasible set**:

```
F = {(x, s) : Ax = b, Cx + s = d, s ≥ 0}
```

This is equivalent to solving:

```
minimize   ‖(x, s) - (x_newton, s_newton)‖²
subject to Ax = b
           Cx + s = d
           s ≥ 0
```

**The key insight:** F is a **convex set** (intersection of an affine subspace and the non-negative orthant for s).

---

## 2. Mathematical Analysis: Does Full Projection Work?

### 2.1 Distance Preservation Property

OPEN-M's analysis relies on (Theorem 3, equation 19):

```
‖x̃_t - x*_t‖ ≤ ‖x_t - x*_t‖
```

For projection onto an **affine subspace**, this follows from the Pythagorean theorem.

For projection onto a **convex set** F, we have a stronger result:

**Theorem (Projection onto Convex Sets):** Let F be a closed convex set. The projection P_F is non-expansive:

```
‖P_F(y) - P_F(z)‖ ≤ ‖y - z‖  for all y, z
```

**Corollary:** If y* ∈ F, then:

```
‖P_F(y) - y*‖ ≤ ‖y - y*‖
```

**Conclusion:** The distance-preservation property DOES hold for projection onto F.

### 2.2 The Projection as a QP

Projecting onto F requires solving a Quadratic Program:

```python
def project_onto_feasible_set(x_curr, s_curr, A, b, C, d):
    """
    Solve: min  0.5 * ||(x,s) - (x_curr, s_curr)||^2
           s.t. Ax = b
                Cx + s = d
                s >= 0

    This is a convex QP with:
    - n + m variables (x and s)
    - p + m equality constraints (from Ax = b and Cx + s = d)
    - m inequality constraints (s >= 0)
    """
    from scipy.optimize import minimize

    n, m, p = len(x_curr), len(s_curr), A.shape[0] if A.size > 0 else 0
    y_curr = np.concatenate([x_curr, s_curr])

    def objective(y):
        return 0.5 * np.sum((y - y_curr)**2)

    def grad(y):
        return y - y_curr

    # Equality constraints: Ax = b, Cx + s = d
    def eq_constraints(y):
        x, s = y[:n], y[n:]
        constraints = []
        if p > 0:
            constraints.append(A @ x - b)
        constraints.append(C @ x + s - d)
        return np.concatenate(constraints) if constraints else np.array([])

    # Inequality constraints: s >= 0 (as -s <= 0)
    bounds = [(None, None)] * n + [(0, None)] * m

    result = minimize(
        objective, y_curr, jac=grad,
        constraints={'type': 'eq', 'fun': eq_constraints},
        bounds=bounds, method='SLSQP'
    )

    return result.x[:n], result.x[n:]
```

**Computational cost:** O(n³) per projection (solving a QP), compared to O(n) for clipping.

---

## 3. Why Full Projection Still Fails

Despite satisfying the distance property, full projection has critical issues:

### 3.1 Active Set Changes Destroy Newton Structure

When s_i = 0 after projection (constraint i becomes active), the problem structure changes:

**Before projection (s > 0):**
- Working in reduced space of dimension n - p (null space of A)
- Newton direction computed assuming interior point

**After projection (some s_i = 0):**
- The effective constraint set has changed
- The reduced Hessian has different dimension
- The Newton direction from the pre-projection point is no longer valid

```python
# Example: How active set changes affect Newton step

# Pre-projection Newton direction (assuming s > 0):
# Solve: [H, A^T, C^T] [dx]   [-grad_x]
#        [A,  0,   0 ] [nu] = [  0    ]
#        [C,  0,   0 ] [la]   [  0    ]

# Post-projection (if s_i = 0 for some i):
# The i-th inequality is now ACTIVE
# Newton step should account for this, but we already computed it!
```

### 3.2 Projection Can Land on Boundary Kinks

The feasible set F has "kinks" where s_i = 0 (faces of the polytope). After projection:

1. You may be on a face where ∇f is discontinuous in the feasible direction
2. The next Newton step may point outside F
3. You need another projection, creating a cycle

This is fundamentally different from OPEN-M's equality-only case where the feasible set is a smooth affine subspace.

### 3.3 Newton Convergence Analysis Breaks Down

OPEN-M's Lemma 3 requires:
- ‖∇²f(x) - ∇²f(x*)‖ ≤ L‖x - x*‖ (Lipschitz Hessian)
- Iterates stay in a neighborhood where Newton contracts

**Problem:** Near the boundary where s_i ≈ 0:
- If using barrier: Hessian blows up (see barrier_reformulation_analysis.md)
- If using projection: The effective Hessian changes discontinuously when constraints activate

```python
# Near boundary, reduced Hessian changes dimension:
#
# s = (0.1, 0.2, 0.3)  ->  H_reduced is (n-p) x (n-p)
# s = (0, 0.2, 0.3)    ->  H_reduced is (n-p-1) x (n-p-1)
#
# This discontinuity breaks Lipschitz Hessian assumption
```

### 3.4 Computational Cost

Each iteration requires:
1. Newton step computation: O(n³)
2. Projection onto F (QP solve): O(n³)

**Total: 2× the computational cost**, and the projection QP may be harder than the original Newton system.

### 3.5 No Guarantee of Staying Interior

Even with perfect projection onto F, the **next Newton step** may exit F again:

```
x_t ∈ F  →  Newton step  →  x_newton ∉ F  →  Project  →  x_proj ∈ F
x_proj ∈ F  →  Newton step  →  x_newton' ∉ F  →  Project  →  ...
```

Every iteration may require projection, and each projection changes the trajectory in ways the analysis doesn't account for.

---

## 4. Numerical Experiment Design

To test whether full projection works in practice, we would implement:

```python
class FeasibleProjectionAlgorithm:
    """
    Algorithm:
    1. Project onto F = {Ax=b, Cx+s=d, s>=0}
    2. Compute Newton step (ignoring s>=0 for now)
    3. Take full Newton step
    4. Project result onto F
    """

    def step(self, instance, x_star, f_star):
        n, m, p = instance.n, instance.m, instance.p

        # Current point
        y = np.concatenate([self.x, self.s])

        # Step 1: Project onto F to handle time-varying constraints
        x_proj, s_proj = project_onto_feasible_set(
            self.x, self.s, instance.A, instance.b,
            instance.F, instance.g
        )

        # Track distance before/after projection
        y_star = np.concatenate([x_star, instance.g - instance.F @ x_star])
        dist_before_proj = np.linalg.norm(y - y_star)
        dist_after_proj = np.linalg.norm(
            np.concatenate([x_proj, s_proj]) - y_star
        )

        # Step 2: Compute Newton step in augmented space
        # Using barrier on s (or treating s>=0 as active constraints)
        grad, hess = self._compute_gradient_hessian(x_proj, s_proj, instance)

        # Step 3: Solve KKT for Newton direction
        dy = self._solve_newton_system(grad, hess, instance)

        # Step 4: Take full Newton step
        x_newton = x_proj + dy[:n]
        s_newton = s_proj + dy[n:]

        # Step 5: Project back onto F
        x_new, s_new = project_onto_feasible_set(
            x_newton, s_newton, instance.A, instance.b,
            instance.F, instance.g
        )

        # Track metrics
        dist_after_newton_proj = np.linalg.norm(
            np.concatenate([x_new, s_new]) - y_star
        )

        # Check if projection helped or hurt
        # (For convex F, should always help or stay same)
        proj_helped = dist_after_newton_proj <= dist_after_proj

        self.x, self.s = x_new, s_new
        return self._compute_metrics(...)
```

### Metrics to Track

1. **Distance preservation:** Does ‖y_proj - y*‖ ≤ ‖y - y*‖ hold? (Should always)
2. **Newton step exits F:** How often does x_newton violate s ≥ 0?
3. **Active set changes:** How many constraints activate/deactivate per step?
4. **Convergence:** Does the algorithm converge to x*?
5. **Constraint satisfaction:** Is Cx ≤ d satisfied (via s ≥ 0)?

### Expected Results

Based on the analysis:
- Distance preservation should hold (convex projection property)
- Newton step will frequently exit F (no line search)
- Active set changes will cause erratic behavior
- Convergence will be worse than pure equality-constrained OPEN-M
- Computational cost will be ~2× higher

---

## 5. Critical Issues with OPEN-M Itself

**Important:** Even if the projection approach worked, OPEN-M has fundamental problems (from the review):

### 5.1 Wrong Hessian Formula (Lemma 2)

The paper writes:
```
∇²f̃_t(z) = F_t ∇²f_t(x*_t) F_t^T  (WRONG - gives n×n matrix)
```

Correct formula:
```
∇²f̃_t(z) = F_t^T ∇²f_t(x) F_t  (gives (n-p)×(n-p) matrix)
```

The final bounds happen to be correct because ‖F_t‖ = 1 is assumed.

### 5.2 Condition 2 Forces v ≈ 0 (Theorem 2)

**This is the most devastating issue.**

The condition requires: v ≤ γ - (2L/h)γ²

With γ = h/(2L), this gives:
```
v ≤ h/(2L) - (2L/h) · h²/(4L²) = 0
```

**No variation in optima is allowed!** The "time-varying" aspect becomes trivial.

### 5.3 Wrong Complexity Claim (Remark 2)

Claims O(n⁵ log n) complexity. Matrix inversion is O(n³).

### 5.4 False Novelty Claims

Claims to be "first online second-order algorithm with constraints" but their own reference [9] (Abernethy, Hazan, Rakhlin 2012) presents interior-point methods for online learning.

### 5.5 Non-Smooth Objective in Experiments

Uses f(x) = α exp(β|x|) which is **not differentiable at x = 0**, violating their own assumptions.

---

## 6. Comparison: Clipping vs Full Projection vs Barrier

| Aspect | Clipping | Full Projection | Barrier (proper IPM) |
|--------|----------|-----------------|---------------------|
| Distance preservation | NO | YES | N/A |
| Maintains Cx + s = d | NO | YES | N/A (different formulation) |
| Maintains s ≥ 0 | YES (by construction) | YES | YES (strict interior) |
| Computational cost/iter | O(n³) | O(n³) × 2 | O(n³) |
| Newton convergence | Breaks | Breaks (active sets) | Works (central path) |
| Handles active constraints | NO | Poorly | Via barrier terms |
| OPEN-M compatible | NO | Partially | NO (needs full IPM) |

---

## 7. Alternative Approaches

If extending OPEN-M to inequalities is truly needed:

### 7.1 Active Set Method (OPEN-M + Active Set QP)

1. Identify active constraints: A_active = {i : C_i x ≈ d_i}
2. Treat active constraints as equalities
3. Apply OPEN-M in the reduced space
4. Update active set when necessary

**Issue:** Active set identification is expensive and discontinuous.

### 7.2 Penalty Method

Add penalty: min f(x) + ρ · ‖max(Cx - d, 0)‖²

**Issue:** Introduces ρ parameter, smooths constraint violations.

### 7.3 Augmented Lagrangian

Combine Lagrangian with quadratic penalty.

**Issue:** Requires dual updates, more complex analysis.

### 7.4 Full Interior Point Method

Follow the central path with barrier parameter μ → ∞.

**Issue:** This is a complete IPM, not a simple extension of OPEN-M.

---

## 8. Conclusion

**Can full projection onto F save the slack variable approach?**

**Partially, but not enough to make it work.**

**What full projection fixes:**
- Distance preservation property holds (convex projection)
- All constraints (Ax = b, Cx + s = d, s ≥ 0) are satisfied

**What full projection cannot fix:**
- Newton convergence analysis breaks at active set changes
- Computational cost doubles
- The underlying OPEN-M algorithm has critical errors (Condition 2 forces v ≈ 0)

**Bottom line:** Full projection onto the feasible set is mathematically cleaner than clipping, but it doesn't resolve the fundamental incompatibility between the Newton-based OPEN-M framework and inequality constraints. The core issue is that **inequality constraints create a polyhedral feasible region where the structure changes discontinuously at constraint boundaries**, while OPEN-M assumes a smooth affine feasible set.

---

## 9. Recommended Next Steps

1. **Implement the full projection approach** to verify the analysis numerically
2. **Track active set changes** to quantify how often constraints activate/deactivate
3. **Compare convergence** between clipping, full projection, and (for reference) a proper IPM
4. **Consider whether the problem class** (online LPs with inequality constraints) truly needs OPEN-M or whether simpler projection methods (e.g., online gradient descent with projection) suffice

---

## References

1. OPEN-M paper: "An Online Newton's Method for Time-varying Linear Equality Constraints"
2. Boyd & Vandenberghe, "Convex Optimization" - Chapters 10-11
3. Nocedal & Wright, "Numerical Optimization" - Active set methods
4. Abernethy, Hazan, Rakhlin (2012) - Online IPM (their reference [9])
