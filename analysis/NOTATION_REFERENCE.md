# Notation Reference for OIPM with Inequality Constraints

## Problem Parameters

### Original Problem Dimensions
- $n$ = number of decision variables
- $m$ = number of equality constraints
- $p$ = number of inequality constraints

### Time Horizon
- $T$ = number of time steps
- $t \in \{1, 2, \ldots, T\}$ = current time step

### Matrices and Vectors
- $c \in \mathbb{R}^n$ = cost vector (fixed over time)
- $A \in \mathbb{R}^{m \times n}$ = equality constraint matrix (fixed)
- $F \in \mathbb{R}^{p \times n}$ = inequality constraint matrix (fixed)
- $b_t \in \mathbb{R}^m$ = equality constraint RHS at time $t$ (time-varying)
- $g_t \in \mathbb{R}^p$ = inequality constraint RHS at time $t$ (time-varying)

---

## Decision Variables

### Original Problem
- $x_t \in \mathbb{R}^n$ = decision variable produced by algorithm at time $t$
- $x_t^* \in \mathbb{R}^n$ = optimal solution to original problem at time $t$

### Slack Variables
- $s_t \in \mathbb{R}^p$ = slack variables at time $t$, where $s_{t,i} > 0$ for all $i$
- $s_t^* \in \mathbb{R}^p$ = optimal slack variables at time $t$
- Relationship: $s_t^* = g_t - F x_t^*$

### Augmented Variables
- $z_t = \begin{bmatrix} x_t \\ s_t \end{bmatrix} \in \mathbb{R}^{n+p}$ = augmented variable at time $t$
- $z_t^* = \begin{bmatrix} x_t^* \\ s_t^* \end{bmatrix} \in \mathbb{R}^{n+p}$ = optimal augmented variable at time $t$

---

## Path Variation Measures

These are the **key quantities** that measure how much things change over time.

### Definition Template
For any sequence of vectors $\{v_t\}_{t=1}^T$, the **path variation** is:
$$V^v = \sum_{t=1}^T \|v_t - v_{t-1}\|$$

**Convention:** We set $v_0 = v_1$ so the $t=1$ term contributes zero.

### Specific Path Variations

#### 1. Original Variable Path Variation
$$\boxed{V_T^x = \sum_{t=1}^T \|x_t^* - x_{t-1}^*\|}$$

**Meaning:** Total distance traveled by the optimal solution in the original variable space.

**Interpretation:** 
- Measures how much the optimal $x$ changes as constraints vary
- If environment is "slowly varying", $V_T^x$ is small
- If environment changes drastically, $V_T^x$ is large

---

#### 2. Slack Variable Path Variation
$$\boxed{V_T^s = \sum_{t=1}^T \|s_t^* - s_{t-1}^*\|}$$

**Meaning:** Total distance traveled by the optimal slack variables.

**Key Inequality (Slack Coupling):**
$$V_T^s \leq V_g^{\text{ineq}} + \|F\| \cdot V_T^x$$

**Why:** Because $s_t^* = g_t - F x_t^*$, so:
$$s_t^* - s_{t-1}^* = (g_t - g_{t-1}) - F(x_t^* - x_{t-1}^*)$$

Taking norms and summing:
$$\|s_t^* - s_{t-1}^*\| \leq \|g_t - g_{t-1}\| + \|F\| \cdot \|x_t^* - x_{t-1}^*\|$$

---

#### 3. Inequality Constraint Variation
$$\boxed{V_g^{\text{ineq}} = \sum_{t=1}^T \|g_t - g_{t-1}\|}$$

**Meaning:** Total change in the inequality constraint right-hand side.

**Interpretation:**
- Measures how much the inequality constraints $Fx \leq g_t$ shift over time
- Independent of optimal solutions - purely environment changes
- If $g_t$ is constant, $V_g^{\text{ineq}} = 0$

**Convention:** Set $g_0 = g_1$ for consistency.

**Example:**
- If $g_t = g_1$ for all $t$ (static constraints): $V_g^{\text{ineq}} = 0$
- If $g_t$ increments by $\delta$ each step: $V_g^{\text{ineq}} \approx (T-1) \|\delta\|$
- If $g_t$ oscillates: $V_g^{\text{ineq}}$ sums all the oscillation magnitudes

---

#### 4. Equality Constraint Variation
$$\boxed{V_b^{eq} = \sum_{t=1}^T \|b_t - b_{t-1}\|}$$

**Meaning:** Total change in the equality constraint right-hand side.

**Note:** This appears in the **constraint violation** bound but not directly in the **regret** bound (it affects regret indirectly through $V_T^x$).

---

#### 5. Augmented Path Variation
$$\boxed{V_T^{\text{aug}} = \sum_{t=1}^T \|z_t^* - z_{t-1}^*\|}$$

**Meaning:** Path variation in the augmented space $\mathbb{R}^{n+p}$.

**Key Decomposition (Triangle Inequality):**
$$V_T^{\text{aug}} \leq V_T^x + V_T^s$$

**Combining with slack coupling:**
$$V_T^{\text{aug}} \leq V_T^x + V_g^{\text{ineq}} + \|F\| \cdot V_T^x = (1 + \|F\|) V_T^x + V_g^{\text{ineq}}$$

---

## Algorithm Parameters

### Barrier Method
- $\phi(s) = -\sum_{i=1}^p \log(s_i)$ = self-concordant barrier for $s > 0$
- $\nu_f = p$ = complexity parameter of barrier
- $\eta_t > 0$ = barrier weight at time $t$
- $\eta_0 > 0$ = initial barrier weight
- $\beta > 1$ = barrier parameter update rate (typically $\beta \in (1, 1.5]$)

### Update Rule
$$\eta_{t+1} = \beta \cdot \eta_t$$

So $\eta_t = \beta^t \eta_0$ grows exponentially.

---

## Performance Metrics

### Dynamic Regret
$$\boxed{R_d(T) = \sum_{t=1}^T \left[c^T x_t - c^T x_t^*\right]}$$

**Meaning:** Cumulative difference between algorithm's cost and optimal cost.

**Goal:** Show that $R_d(T)$ is $O(1 + V_T^x)$, i.e., bounded by initialization + path variation.

---

### Constraint Violation (Equality Constraints)
$$\text{Vio}(T) = \sum_{t=1}^T \|A x_t - b_t\|$$

**Meaning:** Total amount by which algorithm violates equality constraints.

**Bound:** $\text{Vio}(T) \leq V_b^{eq}$ (tracks constraint changes)

---

## Matrix Norms

We use the **spectral norm** (operator 2-norm) for matrices:
$$\|M\| = \sup_{v \neq 0} \frac{\|Mv\|_2}{\|v\|_2} = \sigma_{\max}(M)$$

where $\sigma_{\max}(M)$ is the largest singular value of $M$.

**Key Property (Submultiplicativity):**
$$\|Mv\| \leq \|M\| \cdot \|v\|$$

This is used to bound $\|F(x_t^* - x_{t-1}^*)\| \leq \|F\| \cdot \|x_t^* - x_{t-1}^*\|$.

---

## The Main Result

### Regret Bound for Original Problem

$$\boxed{R_d(T) \leq \frac{11 p \beta}{5 \eta_0 (\beta - 1)} + \|c\| \left[(1 + \|F\|) V_T^x + V_g^{\text{ineq}}\right]}$$

**Components:**

| Term | Value | Interpretation |
|------|-------|----------------|
| Constant | $\frac{11 p \beta}{5 \eta_0 (\beta - 1)}$ | Cost of initializing barrier (depends on # inequalities $p$) |
| Direct tracking | $\|c\| \cdot V_T^x$ | Cost scales with how much optimal $x$ changes |
| Coupling cost | $\|c\| \cdot \|F\| \cdot V_T^x$ | Additional cost because slacks must track $x$ via $F$ |
| Environment cost | $\|c\| \cdot V_g^{\text{ineq}}$ | Cost scales with how much constraint RHS changes |

**Key Insight:** The factor $(1 + \|F\|)$ on $V_T^x$ captures the coupling between original and slack variables.

---

### Constraint Violation Bound

$$\boxed{\text{Vio}(T) \leq V_b^{eq} + V_g^{\text{ineq}}}$$

**Interpretation:** Algorithm tracks changes in both equality and inequality constraints.

---

## Summary Table: All Notation

| Symbol | Definition | Type | Dimension |
|--------|-----------|------|-----------|
| $n$ | # original variables | Constant | Scalar |
| $m$ | # equality constraints | Constant | Scalar |
| $p$ | # inequality constraints | Constant | Scalar |
| $T$ | Time horizon | Constant | Scalar |
| $c$ | Cost vector | Fixed | $\mathbb{R}^n$ |
| $A$ | Equality matrix | Fixed | $\mathbb{R}^{m \times n}$ |
| $F$ | Inequality matrix | Fixed | $\mathbb{R}^{p \times n}$ |
| $b_t$ | Equality RHS | Time-varying | $\mathbb{R}^m$ |
| $g_t$ | Inequality RHS | Time-varying | $\mathbb{R}^p$ |
| $x_t^*$ | Optimal $x$ at time $t$ | Solution | $\mathbb{R}^n$ |
| $s_t^*$ | Optimal slack at time $t$ | Solution | $\mathbb{R}^p$ |
| $V_T^x$ | Path variation of $x$ | Derived | $\sum_{t=1}^T \|x_t^* - x_{t-1}^*\|$ |
| $V_T^s$ | Path variation of slacks | Derived | $\sum_{t=1}^T \|s_t^* - s_{t-1}^*\|$ |
| $V_g^{\text{ineq}}$ | Inequality RHS variation | Environment | $\sum_{t=1}^T \|g_t - g_{t-1}\|$ |
| $V_b^{eq}$ | Equality RHS variation | Environment | $\sum_{t=1}^T \|b_t - b_{t-1}\|$ |
| $\nu_f$ | Barrier complexity | Constant | $= p$ |
| $\eta_0$ | Initial barrier weight | Parameter | Scalar > 0 |
| $\beta$ | Barrier update rate | Parameter | Scalar > 1 |
| $R_d(T)$ | Dynamic regret | Performance | $\sum_{t=1}^T [c^T x_t - c^T x_t^*]$ |

---

## Important Inequalities

### Triangle Inequality (Vector Norms)
$$\|u + v\| \leq \|u\| + \|v\|$$

Used to decompose $\|z_t^* - z_{t-1}^*\| = \|[x_t^* - x_{t-1}^*; s_t^* - s_{t-1}^*]\|$.

### Submultiplicativity (Matrix-Vector)
$$\|Mv\| \leq \|M\| \cdot \|v\|$$

Used to bound $\|F(x_t^* - x_{t-1}^*)\| \leq \|F\| \cdot \|x_t^* - x_{t-1}^*\|$.

### Slack Coupling Inequality
$$\|s_t^* - s_{t-1}^*\| \leq \|g_t - g_{t-1}\| + \|F\| \cdot \|x_t^* - x_{t-1}^*\|$$

**Proof:** $s_t^* - s_{t-1}^* = (g_t - g_{t-1}) - F(x_t^* - x_{t-1}^*)$, then apply triangle inequality and submultiplicativity.

### Summing Over Time
$$V_T^s = \sum_{t=1}^T \|s_t^* - s_{t-1}^*\| \leq \sum_{t=1}^T \|g_t - g_{t-1}\| + \|F\| \sum_{t=1}^T \|x_t^* - x_{t-1}^*\| = V_g^{\text{ineq}} + \|F\| \cdot V_T^x$$

---

## FAQ

### Q: Why does $V_g^{\text{ineq}}$ appear separately from $V_T^x$?

**A:** Because $g_t$ changes are **exogenous** (determined by the environment), while $x_t^*$ changes are **endogenous** (depend on solving the optimization problem). Both contribute to slack variation via:
$$s_t^* = g_t - F x_t^*$$

### Q: What if $g_t$ is constant?

**A:** Then $V_g^{\text{ineq}} = 0$, and the bound becomes:
$$R_d(T) \leq \frac{11 p \beta}{5 \eta_0 (\beta - 1)} + \|c\| (1 + \|F\|) V_T^x$$

The factor $(1 + \|F\|)$ still appears because slacks must track $x$ changes.

### Q: Can I remove the $(1 + \|F\|)$ factor?

**A:** No! This factor is **tight**. Consider a problem where $F$ has large norm - even small changes in $x$ induce large changes in slacks, and the augmented path variation genuinely increases by this factor.

### Q: What norm should I use for vectors?

**A:** The Euclidean ($\ell^2$) norm: $\|v\| = \sqrt{\sum_i v_i^2}$. This is standard in the OIPM paper.

### Q: What norm should I use for matrices?

**A:** The spectral (operator 2-norm): $\|M\| = \sigma_{\max}(M)$. This ensures submultiplicativity and compatibility with vector norms.

---

## Concrete Numerical Example

Let's compute $V_g^{\text{ineq}}$ for a specific sequence.

### Setup
- $p = 3$ inequality constraints
- $T = 5$ time steps
- Inequality RHS vectors:

| Time $t$ | $g_t$ | 
|----------|-------|
| 0 | $[5.0, 3.0, 4.0]^T$ |
| 1 | $[5.0, 3.0, 4.0]^T$ |
| 2 | $[5.2, 3.1, 4.3]^T$ |
| 3 | $[5.5, 2.9, 4.5]^T$ |
| 4 | $[5.3, 3.0, 4.2]^T$ |
| 5 | $[5.4, 3.2, 4.1]^T$ |

### Computation

**Step 1:** Compute $\|g_t - g_{t-1}\|$ for each $t$:

- $t=1$: $\|g_1 - g_0\| = \|[0, 0, 0]\| = 0$
- $t=2$: $\|g_2 - g_1\| = \|[0.2, 0.1, 0.3]\| = \sqrt{0.04 + 0.01 + 0.09} = \sqrt{0.14} \approx 0.374$
- $t=3$: $\|g_3 - g_2\| = \|[0.3, -0.2, 0.2]\| = \sqrt{0.09 + 0.04 + 0.04} = \sqrt{0.17} \approx 0.412$
- $t=4$: $\|g_4 - g_3\| = \|[-0.2, 0.1, -0.3]\| = \sqrt{0.04 + 0.01 + 0.09} = \sqrt{0.14} \approx 0.374$
- $t=5$: $\|g_5 - g_4\| = \|[0.1, 0.2, -0.1]\| = \sqrt{0.01 + 0.04 + 0.01} = \sqrt{0.06} \approx 0.245$

**Step 2:** Sum over all time steps:

$$V_g^{\text{ineq}} = \sum_{t=1}^5 \|g_t - g_{t-1}\| = 0 + 0.374 + 0.412 + 0.374 + 0.245 = 1.405$$

### Interpretation

- The inequality constraints changed with total variation of **1.405**
- If constraints were static ($g_t = g_1$ for all $t$): $V_g^{\text{ineq}} = 0$
- The larger $V_g^{\text{ineq}}$, the more the environment changes
- This contributes directly to the regret bound: $\|c\| \cdot V_g^{\text{ineq}}$

### How This Appears in Regret Bound

If $\|c\| = 2.0$, then the constraint variation contributes:
$$\|c\| \cdot V_g^{\text{ineq}} = 2.0 \times 1.405 = 2.81$$
to the regret bound.

Compare this to the other component:
- If $V_T^x = 1.0$ and $\|F\| = 3.0$, then:
  $$\|c\| \cdot (1 + \|F\|) \cdot V_T^x = 2.0 \times 4.0 \times 1.0 = 8.0$$

So in this example, the constraint changes contribute $2.81$ while the solution tracking contributes $8.0$ to the path-dependent part of the regret.

---

## Common Mistakes to Avoid

1. **Forgetting $V_g^{\text{ineq}}$ in the bound**
   - Incorrect: $R_d(T) \leq \text{const} + \|c\| (1 + \|F\|) V_T^x$
   - Correct: $R_d(T) \leq \text{const} + \|c\| [(1 + \|F\|) V_T^x + V_g^{\text{ineq}}]$

2. **Confusing $V_g^{\text{ineq}}$ with $V_T^s$**
   - $V_g^{\text{ineq}}$ = how much $g_t$ changes (environment)
   - $V_T^s$ = how much $s_t^*$ changes (solution)
   - They're related by: $V_T^s \leq V_g^{\text{ineq}} + \|F\| V_T^x$

3. **Thinking $(1 + \|F\|)$ multiplies everything**
   - Incorrect: $(1 + \|F\|)(V_T^x + V_g^{\text{ineq}})$
   - Correct: $(1 + \|F\|) V_T^x + V_g^{\text{ineq}}$
   - Reasoning: $V_g^{\text{ineq}}$ appears directly in slack coupling, not multiplied by $\|F\|$

4. **Not setting $v_0 = v_1$ convention**
   - Remember: First term $\|v_1 - v_0\|$ should be zero
   - Set $x_0^* = x_1^*$, $s_0^* = s_1^*$, $g_0 = g_1$, $b_0 = b_1$

---

**Last updated:** December 8, 2024

This notation reference provides precise mathematical definitions for all quantities used in the OIPM inequality extension analysis.
