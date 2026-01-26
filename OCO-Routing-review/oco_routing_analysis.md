# Analysis: Online Convex Optimization for On-Board Routing in High-Throughput Satellites

**Paper:** Bélanger et al., "Online Convex Optimization for On-Board Routing in High-Throughput Satellites," arXiv:2409.01488, September 2024

**Date:** January 2026

---

## Executive Summary

This paper applies εOIPM-TEC (from the flawed OIPM-TEC paper) to satellite packet routing. While the theoretical guarantees cited are invalid due to errors in OIPM-TEC, the **practical algorithm may still work** in their specific application due to:

1. A feedback correction mechanism that handles constraint violations
2. Problem structure that may avoid the worst failure modes
3. Conservative parameter choices (large η = 10⁴)

However, the claimed theoretical guarantees do not hold, and the paper should be understood as presenting a **heuristic with empirical validation**, not a theoretically justified algorithm.

---

## 1. Paper Overview

### 1.1 Problem Setting

The paper considers packet routing in Extremely High-Throughput Satellites (EHTS) with:
- M = 16 modem banks
- P = 3 priority levels (VoIP, messaging, email)
- Time horizon T = 100 steps
- MPC window W = 5 steps

### 1.2 Optimization Problem

Problem (1) is a multi-commodity flow problem:

**Objective (1a):** Minimize packet loss cost
```
min Σ_{τ,p,m} L_p^m(τ) k_p
```

**Equality Constraints:**
- (1b) Packet balance: f_p^{in,m} - f_p^{out,m} - ΔQ_p^m - L_p^m = 0
- (1c) Scheduler normalization: Σ_p w_p^m = 1
- (1d) **TIME-VARYING**: Inflow = demand: Σ_m f_p^{in,m} = F̂_p(t)
- (1e) Queue dynamics: Q_p^m(t+1) = Q_p^m(t) + ΔQ_p^m(t)
- (1f) Initial/final queue: Q_p^m(0) = Q_p^m(T-1) = Q_0

**Inequality Constraints:**
- (1g) Scheduler bounds: 0 ≤ w_p^m ≤ 1
- (1h) Ramp constraint: |w_p^m(t) - w_p^m(t-1)| ≤ Δw̄
- (1i) Outflow limit: f_p^{out,m} ≤ w_p^m / Δs
- (1j) Queue capacity: Σ_p Q_p^m ≤ Q̄
- (1k) Bandwidth limit: Σ_p f_p^{out,m} ≤ C̄^m

### 1.3 Reformulation as Standard Form

Problem (2):
```
min  c^T x_t
s.t. A x_t - b_t = 0     (equality, time-varying via b_t)
     C x_t - d ≤ 0       (inequality, time-invariant)
```

Where:
- x_t ∈ ℝ^{6PMW} contains [f^{in}, w, L, f^{out}, Q, ΔQ] for all (p,m) pairs across the MPC window
- Dimension N = 6(W+1)MP = 6 × 6 × 16 × 3 = 1728

---

## 2. The Core Algorithm: εOIPM-TEC

### 2.1 Algorithm 1 (OCMPC)

```
Input: x_0, η = 10^4
For t = 0, 1, ..., T-1:
    1. Observe F̂_p(t) (predicted demand)
    2. Implement decision w_p^m(t), f_p^{in,m}(t) from x_t
    3. IF Σ_m f_p^{in,m}(t) ≠ F_p(t) for any p:  [FEEDBACK CORRECTION]
           Apply proportional adjustment
    4. Observe outcome and new state
    5. Newton update:
       [x_{t+1}; -] = [x_t; -] - [∇²φ(x_t)  A^T; A  0]^{-1} [∇d_η(x_t); Ax_t - b_t]
```

Where:
- φ(x) = -Σ_i log(d_i - C_i x) is the log-barrier for inequality constraints
- d_η(x) = η c^T x + φ(x) is the barrier-augmented objective

### 2.2 Key Observation: No Line Search

The algorithm takes a **full Newton step** (line 10 / step 5) without line search. This is problematic because:

1. The Newton step can violate inequality constraints: C x_{t+1} > d
2. When this happens, φ(x_{t+1}) = +∞ (barrier undefined)
3. The next Newton step is undefined

---

## 3. The Feedback Correction Mechanism

### 3.1 What It Does

Lines 6-7 of Algorithm 1:
```
if x_t is not such that Σ_{m=1}^M f_p^{in,m}(t) = F_p(t) ∀p then
    Apply the feedback correction
```

**This addresses prediction error**, not Newton step failure.

The optimization uses **predicted** demand F̂_p(t), but the actual demand F_p(t) is observed after the decision. The feedback correction:

1. Compares actual demand F_p(t) to what was allocated: Σ_m f_p^{in,m}(t)
2. Proportionally adjusts inflow allocations to match reality

**Mathematically:**
```
f_p^{in,m}(t) ← f_p^{in,m}(t) × F_p(t) / (Σ_m f_p^{in,m}(t))
```

### 3.2 What It Does NOT Do

The feedback correction does **NOT**:
- Fix Newton steps that exit the feasible region
- Handle inequality constraint violations
- Provide theoretical guarantees

### 3.3 Why This Matters

The paper states (page 3):
> "While εOIPM-TEC is effective in many scenarios, it is only guaranteed to respect equality constraints under conditions more stringent than can be assumed for this application."

This is an implicit acknowledgment that:
1. The theoretical guarantees don't apply to their setting
2. The feedback correction is a practical workaround, not a theoretically justified fix

---

## 4. Dependency on OIPM-TEC

### 4.1 Cited Claims

The paper claims (page 3):
> "OIPM-TEC guarantees time-averaged optimal decisions on inequality-constrained convex problems with time-varying equality constraints"

And (page 3):
> "The sequence {x_t}_{t=1}^T provided by Algorithm 1 has provable bounds on dynamic regret and constraint violations under certain conditions. Interested readers are referred to [10] for further details."

### 4.2 Why These Claims Are Invalid

OIPM-TEC (reference [10]) contains critical errors:

| Lemma/Argument | Error | Impact |
|----------------|-------|--------|
| Lemma invHess | False equality ‖M‖ = ‖M⁻¹‖ | Invalidates Hessian bounds |
| Lemma nred | Unjustified inequality, missing factors | Invalidates Newton reduction |
| Lemma yx | Ignores cross-terms in quadratic form | Invalidates primal-dual relation |
| Barrier complexity | Misapplied Renegar reference | Invalidates barrier bounds |

**The "provable bounds" cited do not actually exist.**

### 4.3 The ε-Tolerance Claim

The paper claims (page 4):
> "ε ~ O(N)/η, where N = 6(W+1)MP is the dimension"

With N = 1728 and η = 10⁴, this gives ε ≈ 0.17.

**Problem:** This bound relies on OIPM-TEC Theorem 2, which depends on the flawed lemmas. The ε-tolerance is not proven.

---

## 5. Analysis of Their Specific Problem

### 5.1 Problem Structure

Their problem has special structure that may help:

1. **Linear objective**: c^T x (no curvature from objective)
2. **Linear constraints**: Both equality and inequality constraints are linear
3. **Box + linear inequalities**: Constraints (1g)-(1k) are relatively simple

### 5.2 The Barrier Hessian

For their problem, the barrier Hessian is:
```
∇²φ(x) = Σ_i C_i^T C_i / (d_i - C_i x)²
```

This is positive semidefinite (good) but:
- Blows up near constraint boundaries
- Has condition number O(1/(min slack)²)

### 5.3 Why Their Problem Might Work Anyway

**Conservative parameter choice:** η = 10⁴ is large, but for a linear program:
- The barrier objective is d_η(x) = η c^T x + φ(x)
- Large η means the linear term dominates
- The barrier optimum is close to the LP optimum

**Wait - this is backwards.** In standard IPM:
- Small η (≈ 1) at the start: barrier dominates, stay away from boundary
- Large η at the end: objective dominates, approach LP optimum

Starting with large η = 10⁴ means:
- The algorithm tries to go directly to the LP optimum
- The LP optimum may be on the constraint boundary
- The barrier Hessian could be extremely ill-conditioned

### 5.4 Potential Saving Grace: Interior Optimal Solutions

If the LP optimal solution lies in the **strict interior** of the inequality constraints (all slacks > 0), then:
- The barrier optimum ≈ LP optimum for large η
- The barrier Hessian is well-conditioned
- Newton steps stay inside the feasible region

**When does this happen?** For their packet routing problem:
- If queues are not at capacity (Q < Q̄)
- If bandwidth is not saturated (outflow < C̄)
- If scheduler weights are not extreme (0 < w < 1)

**In practice:** The simulation with Poisson traffic (λ ∈ {20, 25, 30}) and queue capacity Q̄ = 10 might keep the problem in the interior most of the time.

---

## 6. Numerical Results Analysis

### 6.1 Simulation Setup

| Parameter | Value | Notes |
|-----------|-------|-------|
| T | 100 | Time horizon |
| W | 5 | MPC window |
| M | 16 | Modem banks |
| P | 3 | Priorities |
| η | 10⁴ | Barrier parameter |
| Q̄ | 10 | Queue capacity |
| λ | {20, 25, 30} | Packet arrival rates |

### 6.2 Results Summary

| Method | Packet Loss Cost | vs Hindsight |
|--------|------------------|--------------|
| Batch with hindsight | baseline | 0% |
| MPC (optimal) | +1.24% | — |
| OCMPC | +19.73% | +17.91% vs MPC |
| Proportional | +49.27% | — |

### 6.3 Are These Results Legitimate?

**Yes, as empirical evidence.** The numerical results show:
1. OCMPC performs reasonably (within 20% of optimal)
2. It significantly outperforms the naive proportional method
3. The 100 Monte Carlo runs provide statistical significance

**However:**
1. The results don't validate the theoretical claims (which are invalid)
2. We don't see constraint violation statistics
3. The specific traffic model might keep the problem "easy"

### 6.4 What's Missing from the Numerical Analysis

The paper does **NOT** report:
1. **Inequality constraint violations**: Did C x_t ≤ d hold?
2. **Equality constraint violations**: How bad was Ax_t - b_t before feedback correction?
3. **Queue overflow events**: How often did Q > Q̄?
4. **Newton step sizes**: Were steps small enough to stay feasible?
5. **Hessian condition numbers**: How ill-conditioned was the problem?
6. **Sensitivity to η**: What happens with η = 10², 10³, 10⁵?

---

## 7. What Could Go Wrong

### 7.1 High Traffic Scenarios

If λ increases significantly:
- Queues approach capacity (Q → Q̄)
- Bandwidth approaches saturation
- Barrier Hessian becomes ill-conditioned
- Newton steps may exit feasible region

### 7.2 Active Constraint Scenarios

If the LP optimal solution has active constraints:
- Scheduler weight at boundary: w = 0 or w = 1
- Queue at capacity: Q = Q̄
- Bandwidth saturated: f^{out} = C̄

Then:
- The barrier optimum is pushed away from the LP optimum
- The gap is O(1/η) = O(10⁻⁴) per constraint
- Could accumulate over time

### 7.3 Numerical Instability

With N = 1728 variables and potentially ill-conditioned Hessian:
- Matrix inversion (∇²φ)⁻¹ could be unstable
- Small eigenvalues lead to large Newton steps
- Numerical errors could compound

---

## 8. Comparison with Known Failure Modes

### 8.1 vs. Slack Variable Projection (from our codebase)

The slack projection approach fails because clipping slack variables destroys equality constraints.

**This paper's approach:** Uses log-barrier, no explicit slack variables. The feedback correction handles equality constraint (1d) violations, but doesn't clip.

**Verdict:** Different failure mode. The barrier approach could still fail by exiting the feasible region.

### 8.2 vs. Barrier Method (from our codebase)

Our barrier method experiments show that full Newton steps can exit the feasible region.

**This paper's approach:** Same algorithm structure (full Newton, no line search).

**Key difference:** They have a feedback correction for prediction error, but NOT for Newton step violations.

**Verdict:** Same fundamental vulnerability. Whether it manifests depends on problem conditioning.

---

## 9. Summary of Issues

### 9.1 Theoretical Issues

| Issue | Severity | Notes |
|-------|----------|-------|
| OIPM-TEC proofs are invalid | CRITICAL | Core theoretical foundation is broken |
| ε-tolerance bound unproven | HIGH | Claimed O(N)/η bound not justified |
| No line search | HIGH | Newton step can exit feasible region |
| Barrier Hessian ill-conditioning | MEDIUM | Depends on problem structure |

### 9.2 What Saves Them (Partially)

| Factor | How It Helps |
|--------|--------------|
| Feedback correction | Handles prediction error (but not Newton failure) |
| Interior optimal solutions | If LP optimum is interior, barrier works well |
| Moderate traffic load | λ = 20-30 with Q̄ = 10 keeps problem "easy" |
| Empirical validation | Results show it works in practice |

### 9.3 Open Questions

1. **Does the Newton step ever exit the feasible region?** (Not reported)
2. **What is the actual constraint violation?** (Not reported)
3. **How does performance degrade with higher traffic?** (Not tested)
4. **What happens with different η values?** (Not tested)

---

## 10. Conclusion

### 10.1 For Practitioners

The OCMPC algorithm may work well for satellite routing problems with:
- Moderate traffic loads (not near capacity)
- Interior optimal solutions (no active inequality constraints)
- Short time horizons (errors don't accumulate)

**Do NOT rely on the theoretical guarantees.** They are based on flawed proofs.

### 10.2 For Researchers

The paper should be cited as:
- An application of online optimization to satellite routing
- A heuristic with empirical validation

It should **NOT** be cited for:
- Theoretical guarantees on regret or constraint violation
- Correctness of OIPM-TEC

### 10.3 Recommended Follow-up

1. **Test constraint violations**: Run the algorithm and measure max(Cx - d, 0)
2. **Stress test**: Increase traffic to see when the algorithm fails
3. **Add line search**: Standard IPM fix for feasibility preservation
4. **Compare to proper IPM**: Use MOSEK or similar with warm-starting

---

## Appendix A: Critical Issue — The Barrier Parameter Choice

### A.1 Standard IPM Approach

In standard interior-point methods (Boyd & Vandenberghe, Chapter 11):

1. **Start with small t** (their notation): The barrier term dominates, keeping iterates in the interior
2. **Gradually increase t**: Follow the "central path" towards the optimum
3. **At each t**: Use Newton's method with **line search** to find the barrier optimum

The central path x*(t) satisfies:
- Strictly feasible: Cx*(t) < d for all finite t
- Approaches the constrained optimum: x*(t) → x* as t → ∞

### A.2 The Paper's Approach

The OCO paper uses:
- d_η(x) = η c^T x + φ(x) where φ(x) = -Σ log(d_i - C_i x)
- Fixed η = 10^4 from the start
- No line search

**Large η = 10^4 means:**
- The linear objective (η c^T x) dominates the barrier (φ(x))
- The barrier optimum x*(η) is very close to the LP optimum x*
- If the LP optimum has active constraints, x*(η) is near the boundary
- Near the boundary, the barrier Hessian has entries O((d - Cx)^{-2}) → large

**This is backwards!** You should start with small η and increase it, not start large.

### A.3 Numerical Consequence

With η = 10^4 and slacks s_i = d_i - C_i x:
- If s_i ≈ 0.01 (1% slack), the barrier Hessian term is O(10^4)
- If s_i ≈ 0.001 (0.1% slack), the barrier Hessian term is O(10^6)

The KKT matrix becomes extremely ill-conditioned when any constraint is nearly active.

### A.4 Why They Might Get Away With It

For their specific problem (packet routing):
- Queue capacity Q̄ = 10 packets, arrival rate λ ≤ 30
- If the system is not overloaded, queues stay well below capacity
- This keeps slacks s_i = Q̄ - Q reasonably large
- The barrier Hessian stays reasonably conditioned

**But:** Under high load or congestion, queues approach capacity, slacks shrink, and the algorithm could fail.

---

## Appendix B: Initialization Issues

### B.1 The Paper's Claim

From page 4:
> "We initialize the initial guess x_0 randomly and verify its feasibility before applying the Newton step."

### B.2 What "Feasibility" Means Here

For the barrier method to work, x_0 must satisfy:
1. **Equality constraints**: Ax_0 = b_0 (or close enough)
2. **Strict inequality constraints**: Cx_0 < d (strict interior)

**Problem:** Random initialization is unlikely to satisfy both:
- Random x_0 probably doesn't satisfy Ax_0 = b_0
- Even if projected onto {Ax = b_0}, might not satisfy Cx < d

### B.3 What They Probably Do

They likely:
1. Generate random x_0
2. Project onto equality constraints: x_0 ← x_0 + A^T(AA^T)^{-1}(b_0 - Ax_0)
3. Check Cx_0 < d; if not, regenerate

This could work for their problem where the feasible region is large (low traffic), but becomes problematic when the feasible region shrinks (high traffic).

### B.4 Missing Convergence Guarantee

OIPM-TEC's convergence requires: ‖x_0 - x*_0‖ ≤ γ for some convergence radius γ.

Random initialization provides no guarantee that x_0 is close to the optimum x*_0. The algorithm might take many steps to converge, or diverge entirely.

---

## Appendix C: The Feedback Correction in Detail

### C.1 When Is It Triggered?

The correction triggers when actual demand F_p(t) differs from the sum of allocated inflows Σ_m f_p^{in,m}(t).

This happens because:
1. The optimization used predicted demand F̂_p(t) from the MMPP model
2. The actual realization F_p(t) is a Poisson random variable
3. F_p(t) ≠ F̂_p(t) almost surely

### C.2 The Proportional Adjustment

The paper says "proportionally adjusts the flow allocation." This likely means:

```
For each priority p:
    total_allocated = Σ_m f_p^{in,m}(t)
    actual_demand = F_p(t)
    ratio = actual_demand / total_allocated

    For each modem m:
        f_p^{in,m}(t) ← f_p^{in,m}(t) × ratio
```

### C.3 Does This Fix Constraint Violations?

**For constraint (1d):** Yes, by construction: Σ_m f_p^{in,m}(t) = F_p(t) after correction.

**For constraint (1b):** Maybe. If inflow changes but outflow/queue/loss don't, the balance equation is violated.

**For constraints (1g)-(1k):** No. The correction doesn't touch scheduler weights, queue states, or bandwidth constraints.

### C.4 What the Authors Acknowledge

From page 3:
> "While εOIPM-TEC is effective in many scenarios, it is only guaranteed to respect equality constraints under conditions more stringent than can be assumed for this application."

This is a diplomatic way of saying: **the theory doesn't apply, so we added a heuristic fix.**
