# Why OPEN-M Cannot Handle Time-Varying Inequality Constraints

OPEN-M works for time-varying **equality** constraints by projecting onto the constraint manifold and taking a full Newton step. Two natural approaches to extend it to **inequality** constraints both fail.

---

## 1. Slack Variable Projection

**Idea:** Reformulate Fx ≤ g as equality Fx + s = g with s ≥ 0, then apply OPEN-M to the augmented system.

**Algorithm:**
1. Augment variables: y = [x; s]
2. Project onto {y : Fx + s = g} when g changes
3. Take full Newton step
4. Clip s to s ≥ 0 (to maintain slack non-negativity)

**Why it fails:**

| Issue | Explanation |
|-------|-------------|
| **Clipping destroys equality** | After clipping s → max(s, 0), the equality Fx + s = g no longer holds. The algorithm loses track of where it is relative to the constraint. |
| **Clipping can increase distance to optimum** | The optimal s* may have some components = 0 (active constraints). Clipping negative s to 0 doesn't move toward s* in general—it can move away. |
| **Information loss** | Negative slack s < 0 means "how much we're violating Fx ≤ g." Clipping to 0 destroys this information, preventing corrective action. |
| **No convergence guarantee** | OPEN-M's analysis requires staying on the constraint manifold. Clipping breaks this, and violations accumulate over time. |

**Numerical evidence:**
- Equality Fx+s=g violated after clipping (max violation ~0.04)
- Cumulative violations grow over time (no O(V_T) bound)
- Clipping increased distance to optimum in some timesteps

---

## 2. Barrier Reformulation

**Idea:** Replace Fx ≤ g with a log barrier, then apply OPEN-M to the smooth unconstrained problem.

**Algorithm:**
1. Solve: min c'x - (1/μ)∑log(gᵢ - Fᵢx) subject to Ax = b
2. Take full Newton step (no line search, like OPEN-M)
3. Repeat as g changes over time

**Why it fails:**

| Issue | Explanation |
|-------|-------------|
| **Full Newton step exits feasible region** | Standard IPM uses line search to ensure Fx < g. OPEN-M takes full Newton steps, which can overshoot into Fx > g (infeasible). |
| **Hessian conditioning catastrophe** | Barrier Hessian ∝ 1/(g-Fx)². Near the boundary, condition number → ∞, violating OPEN-M's bounded Hessian assumption. |
| **Lipschitz Hessian fails** | OPEN-M requires Lipschitz continuous Hessian. Barrier Hessian has unbounded derivatives near boundary: ∂H/∂x ∝ 1/(g-Fx)³. |
| **Barrier gap** | Barrier optimum ≠ true LP optimum. Gap is O(m/μ), creating persistent tracking error even if algorithm doesn't crash. |

**Numerical evidence:**
- Newton step exits feasible region for μ ≥ 10 (crashes at t=0)
- Hessian condition numbers reach 10¹² (capped in plots)
- Average distance to true optimum: 1.85 (μ=1) to ∞ (crashed)

---

## The Core Problem

OPEN-M's theoretical guarantees rely on:
1. **Staying on the constraint manifold** (equality constraints)
2. **Bounded, Lipschitz Hessian** (for Newton step analysis)
3. **Full Newton step convergence** (no line search needed)

Inequality constraints break all three:
- Slack clipping leaves the augmented equality manifold
- Barrier Hessian is unbounded near the boundary
- Full Newton steps can exit the feasible region

---

## Conclusion

Extending OPEN-M to time-varying inequalities requires fundamentally different techniques—not just reformulating inequalities as equalities or adding barriers. The online setting's "no line search" requirement is incompatible with the geometric structure of inequality constraints.

---

## Experimental Results Summary

### Slack Variable Projection

| Metric | Simple 2D | Medium n=10 |
|--------|-----------|-------------|
| Cumulative regret | 34.4 | 460.6 |
| Clipping hurt distance | 1 timestep | 1 timestep |
| Max Fx+s=g violation | 0.044 | 0.016 |
| Total violation | 0.36 | 0.17 |

### Barrier Method

| Setting | 2D | Medium |
|---------|-----|--------|
| μ=1.0 | survives, dist=1.85 | survives, dist=19.5 |
| μ=10.0 | survives, dist=0.30 | **crashes t=0** |
| μ=100.0 | **crashes t=0** | **crashes t=0** |
| Max condition | 8.0 (μ=10) | 10¹² (capped) |

### Total Variation (Sub-linear)

Both test problems use decaying perturbations to ensure V_T = O(√T):

| T | V_T (2D) | V_T/√T |
|---|----------|--------|
| 10 | 0.36 | 0.115 |
| 50 | 1.06 | 0.149 |
| 100 | 1.57 | 0.157 |
| 500 | 3.75 | 0.168 |
