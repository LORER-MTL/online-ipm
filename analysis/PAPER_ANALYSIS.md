# Analysis of "Online Interior Point Methods for Time-Varying Equality Constraints"

## Paper's Problem Formulation

The paper considers online convex optimization problems:

```
At time t:
minimize    c^T x
subject to  g_i(x) ≤ 0  for i = 1,...,m  (inequality constraints)
            Ax = b_t                       (time-varying equality constraints)
```

Using the barrier method, this becomes:
```
minimize    d_t(x, η) = η·c^T x + ϕ(x)
subject to  Ax = b_t
```

Where:
- `ϕ(x)` is a self-concordant barrier function for the inequality constraints
- `η > 0` is the barrier parameter
- The Hessian `D(x) = ∇²ϕ(x)` is positive definite in the interior

## Key Notation (from the paper)

- `x*_t`: optimal solution at time t for the original problem
- `x^η_t`: optimal solution for the barrier problem with parameter η at time t
- `V_T = Σ ||x*_{t-1} - x*_t||`: **path variation of optimal solutions**
- `V_b = Σ ||b_t - b_{t-1}||`: **constraint variation**
- `c = ||c||`: norm of the objective vector
- `v_f`: barrier parameter (complexity of the barrier function)

## Claimed Bounds (Theorem 1)

**Dynamic Regret:**
```
R_d(T) ≤ (11v_f β)/(5η_0(β-1)) + c·V_T
```

**Constraint Violation:**
```
Vio(T) ≤ V_b
```

Where β > 1 is the barrier parameter update rate.

## Linear Programming as a Special Case

For linear programs with box constraints:
```
minimize    c^T x
subject to  x ≥ 0        (handled by logarithmic barrier)
            Ax = b_t
```

The barrier function is:
```
ϕ(x) = -Σ log(x_i)
```

Properties:
- `∇²ϕ(x) = diag(1/x₁², 1/x₂², ..., 1/xₙ²)` (positive definite)
- This is a self-concordant barrier with `v_f = n`
- All the paper's theory applies to this case

## Paper's Key Assumptions

From the paper's analysis:

1. **Bounded constraint changes** (Lemma 5): `||b_t - b_{t-1}|| ≤ √(3m)/(160)` for stability
2. **Initial feasibility**: Starting point satisfies `||n_0(y_0, η_0)||_D(y_0) ≤ 1/9`
3. **Barrier parameter bound**: `β ≤ 1 + 1/(8√v_f)`
4. **Self-concordant barriers** (Assumption implicit in Lemmas 1-8)

## Regret Bound Analysis

The paper's regret bound is **O(V_T)**, not O(√(V_T·T)). This means:

- If V_T = O(√T), then regret is O(√T) ✓ sublinear
- If V_T = O(T), then regret is O(T) - only linear!
- The bound is tight to V_T with an additive constant

**Critical observation:** The regret bound quality depends entirely on V_T. If the environment is adversarial and forces optimal solutions to change significantly, V_T can be large and the regret bound becomes weak.

## Constraint Violation Analysis

The paper claims `Vio(T) ≤ V_b`, meaning constraint violations are bounded by the total constraint variation.

This is a **strong claim** because:
- It suggests the algorithm "tracks" constraint changes perfectly
- No √T factor - purely linear in constraint variation
- Seems potentially too optimistic for realistic adaptation rates
