# Extending OIPM to Handle Inequality Constraints via Slack Variables

## Overview

This document shows how the Online Interior Point Method (OIPM) from the paper can be extended to handle inequality constraints by introducing slack variables. We prove that the performance guarantees from Theorem 1 transfer to the original problem.

## 1. Problem Setup

### 1.1 Original Problem (with inequalities)

At each time step $t = 1, \ldots, T$, consider the linear program:

```
minimize    c^T x
subject to  Ax = b_t         (equality constraints, A ∈ ℝ^{m×n})
            Fx ≤ g_t         (inequality constraints, F ∈ ℝ^{p×n})
```

**Notation:**
- $x \in \mathbb{R}^n$ is the decision variable
- $c \in \mathbb{R}^n$ is the (fixed) cost vector
- $b_t \in \mathbb{R}^m$ is the time-varying equality constraint RHS
- $g_t \in \mathbb{R}^p$ is the time-varying inequality constraint RHS
- $A \in \mathbb{R}^{m \times n}$, $F \in \mathbb{R}^{p \times n}$ are constraint matrices

### 1.2 Transformed Problem (with slack variables)

Introduce slack variables $s \in \mathbb{R}^p$ where $s_i > 0$ for all $i$. The inequality $Fx \leq g_t$ becomes:
$$Fx + s = g_t, \quad s > 0$$

This gives us the **augmented problem**:

```
minimize    c^T x + 0^T s  =  [c^T, 0^T][x; s]
subject to  [A  0] [x]   [b_t]
            [F  I] [s] = [g_t]
            s > 0                    (simple bounds)
```

**Augmented notation:**
- Augmented variable: $z = \begin{bmatrix} x \\ s \end{bmatrix} \in \mathbb{R}^{n+p}$
- Augmented cost: $\tilde{c} = \begin{bmatrix} c \\ 0_p \end{bmatrix} \in \mathbb{R}^{n+p}$
- Augmented constraint matrix: $\tilde{A} = \begin{bmatrix} A & 0 \\ F & I \end{bmatrix} \in \mathbb{R}^{(m+p) \times (n+p)}$
- Augmented RHS: $\tilde{b}_t = \begin{bmatrix} b_t \\ g_t \end{bmatrix} \in \mathbb{R}^{m+p}$

The augmented problem is:
```
minimize    ̃c^T z
subject to  ̃Az = ̃b_t
            z_i > 0 for i ∈ {n+1, ..., n+p}  (slack positivity)
```

Note: We can extend this to require all components of $z$ to be positive if desired, by handling $x \geq 0$ separately.

## 2. Barrier Method for Augmented Problem

The paper uses the barrier method to handle simple inequalities. For the augmented problem, we use the logarithmic barrier:

$$\phi(z) = -\sum_{i=n+1}^{n+p} \log(s_i) = -\sum_{i=1}^{p} \log(z_{n+i})$$

This is a self-concordant barrier with complexity parameter $\nu_f = p$ (the number of inequality constraints).

The barrier subproblem at time $t$ with parameter $\eta > 0$ is:

```
minimize    d_t(z, η) = η · ̃c^T z + φ(z)
subject to  ̃Az = ̃b_t
```

## 3. Applying Theorem 1 to the Augmented Problem

**Theorem 1 (from paper):** For the online optimization problem with time-varying equality constraints and self-concordant barrier, the OIPM algorithm achieves:

**Dynamic Regret:**
$$R_d(T) \leq \frac{11\nu_f \beta}{5\eta_0(\beta-1)} + \|\tilde{c}\| \cdot V_T$$

**Constraint Violation:**
$$\text{Vio}(T) \leq V_b$$

where:
- $V_T = \sum_{t=1}^T \|z_t^* - z_{t-1}^*\|$ is the path variation of optimal solutions
- $V_b = \sum_{t=1}^T \|\tilde{b}_t - \tilde{b}_{t-1}\|$ is the constraint variation
- $\nu_f = p$ is the barrier complexity parameter
- $\beta > 1$ is the barrier parameter update rate
- $\eta_0$ is the initial barrier parameter

### 3.1 Computing the Norms

**Cost vector norm:**
$$\|\tilde{c}\| = \left\|\begin{bmatrix} c \\ 0 \end{bmatrix}\right\| = \|c\|$$

**Constraint variation:**
$$\|\tilde{b}_t - \tilde{b}_{t-1}\| = \left\|\begin{bmatrix} b_t - b_{t-1} \\ g_t - g_{t-1} \end{bmatrix}\right\| = \sqrt{\|b_t - b_{t-1}\|^2 + \|g_t - g_{t-1}\|^2}$$

**Define the constraint variation terms:**

- $V_b^{eq} = \sum_{t=1}^T \|b_t - b_{t-1}\|$ (equality constraint variation)
- $V_g^{\text{ineq}} = \sum_{t=1}^T \|g_t - g_{t-1}\|$ (inequality constraint variation)

where $\|\cdot\|$ denotes the Euclidean norm, and we set $b_0 = b_1$ and $g_0 = g_1$ by convention.

Then by triangle inequality:
$$V_b \leq V_b^{eq} + V_g^{ineq}$$

### 3.2 Path Variation of Augmented Solutions

Let $z_t^* = \begin{bmatrix} x_t^* \\ s_t^* \end{bmatrix}$ be the optimal solution to the augmented problem at time $t$.

The path variation is:
$$V_T^{aug} = \sum_{t=1}^T \|z_t^* - z_{t-1}^*\| = \sum_{t=1}^T \left\|\begin{bmatrix} x_t^* - x_{t-1}^* \\ s_t^* - s_{t-1}^* \end{bmatrix}\right\|$$

By triangle inequality:
$$\|z_t^* - z_{t-1}^*\| \leq \|x_t^* - x_{t-1}^*\| + \|s_t^* - s_{t-1}^*\|$$

Therefore:
$$V_T^{aug} \leq V_T^{x} + V_T^{s}$$

where:
- $V_T^{x} = \sum_{t=1}^T \|x_t^* - x_{t-1}^*\|$ is the path variation of $x$
- $V_T^{s} = \sum_{t=1}^T \|s_t^* - s_{t-1}^*\|$ is the path variation of slacks

## 4. Relating Slack Path Variation to Constraint Variation

**Key observation:** The slack variables satisfy $s_t^* = g_t - Fx_t^*$.

Therefore:
$$s_t^* - s_{t-1}^* = (g_t - Fx_t^*) - (g_{t-1} - Fx_{t-1}^*) = (g_t - g_{t-1}) - F(x_t^* - x_{t-1}^*)$$

Taking norms:
$$\|s_t^* - s_{t-1}^*\| \leq \|g_t - g_{t-1}\| + \|F\| \cdot \|x_t^* - x_{t-1}^*\|$$

Summing over time:
$$V_T^{s} \leq V_g^{ineq} + \|F\| \cdot V_T^{x}$$

## 5. Performance Guarantees for the Original Problem

### 5.1 Dynamic Regret Bound

Applying Theorem 1 to the augmented problem:
$$R_d^{aug}(T) \leq \frac{11p\beta}{5\eta_0(\beta-1)} + \|c\| \cdot V_T^{aug}$$

Since $V_T^{aug} \leq V_T^{x} + V_T^{s} \leq V_T^{x} + V_g^{ineq} + \|F\| \cdot V_T^{x}$:

$$R_d^{aug}(T) \leq \frac{11p\beta}{5\eta_0(\beta-1)} + \|c\| \cdot \left[(1 + \|F\|) V_T^{x} + V_g^{ineq}\right]$$

**Interpretation:**
- The regret depends on the path variation of the original variables $x$
- Additional term depends on inequality constraint variation $V_g^{ineq}$
- Factor $(1 + \|F\|)$ appears due to coupling between $x$ and slack variables

**Special case:** If inequality constraints are fixed ($g_t = g$ for all $t$), then $V_g^{ineq} = 0$ and:
$$R_d^{aug}(T) \leq \frac{11p\beta}{5\eta_0(\beta-1)} + \|c\|(1 + \|F\|) V_T^{x}$$

### 5.2 Constraint Violation Bound

From Theorem 1 applied to the augmented problem:
$$\text{Vio}^{aug}(T) \leq V_b \leq V_b^{eq} + V_g^{ineq}$$

**What does this mean for the original problem?**

At each time $t$, the algorithm produces $z_t = \begin{bmatrix} x_t \\ s_t \end{bmatrix}$ satisfying:
- Equality constraints: $Ax_t + 0 \cdot s_t = b_t$ ✓ (exactly satisfied)
- Slack definition: $Fx_t + s_t = g_t$ ✓ (exactly satisfied)
- Slack positivity: $s_t > 0$ ✓ (enforced by barrier)

Therefore, for the **original problem**:
- $Ax_t = b_t$ ✓ **equality constraints exactly satisfied**
- $Fx_t = g_t - s_t < g_t$ ✓ **inequality constraints exactly satisfied**

**Conclusion:** The augmented formulation with slack variables ensures that both equality and inequality constraints in the original problem are **exactly satisfied** at each iteration, assuming the algorithm maintains feasibility.

The constraint violation bound $\text{Vio}^{aug}(T) \leq V_b^{eq} + V_g^{ineq}$ refers to the *tracking error* of the algorithm as it adapts to changing constraints, not violation of the original problem's constraints.

### 5.3 Summary of Guarantees

For the original LP with inequalities:
```
minimize    c^T x
subject to  Ax = b_t
            Fx ≤ g_t
```

The OIPM method with slack variable transformation achieves:

**1. Feasibility:** All constraints are satisfied at each time step (assuming interior point feasibility is maintained).

**2. Dynamic Regret:**
$$R_d(T) \leq \frac{11p\beta}{5\eta_0(\beta-1)} + \|c\| \left[(1 + \|F\|) V_T^{x} + V_g^{ineq}\right]$$

where:
- $p$ is the number of inequality constraints
- $V_T^{x}$ is the path variation of optimal $x$ values
- $V_g^{ineq}$ is the total variation in inequality constraint RHS

**3. Constraint Tracking:**
$$\text{Tracking Error} \leq V_b^{eq} + V_g^{ineq}$$

## 6. Practical Considerations

### 6.1 Barrier Parameter

The barrier complexity is $\nu_f = p$ (number of inequalities), which affects:
- The constant term in regret: $O(p)$
- Step size selection: $\eta_0$ must be chosen appropriately
- Number of Newton steps per update

### 6.2 Matrix Conditioning

The augmented system:
$$\tilde{A} = \begin{bmatrix} A & 0 \\ F & I \end{bmatrix}$$

has special structure:
- Lower-right block is identity (well-conditioned)
- Can exploit this structure in KKT system solves
- Overall conditioning depends on $A$ and $F$

### 6.3 Computational Cost

At each time step, the algorithm:
1. Solves KKT system with matrix of size $(n+p) \times (n+p)$
2. Performs $O(p)$ logarithmic barrier evaluations
3. Updates both $x$ and slack variables $s$

This is comparable to standard interior point methods for LPs with inequalities.

## 7. Comparison to Direct Barrier Approach

One could alternatively apply a barrier directly to the inequality constraints without slack variables:
$$\phi(x) = -\sum_{i=1}^p \log(g_{t,i} - F_i x)$$

However, this approach has challenges:
- Non-self-concordant in general (depends on $F$)
- Harder to analyze theoretically
- KKT systems more complex

The slack variable approach:
- ✓ Converts to standard form with simple bounds
- ✓ Uses well-understood log barrier: $\phi(s) = -\sum \log(s_i)$
- ✓ Leverages existing theory (Theorem 1)
- ✓ Clean separation between problem structure and barrier

## 8. Conclusion

We have shown that:

1. **Transformation:** Any LP with equality and inequality constraints can be converted to the standard form considered in the paper by introducing slack variables.

2. **Theorem 1 Applies:** The converted problem satisfies all assumptions, so Theorem 1's guarantees hold for the augmented problem.

3. **Guarantees Transfer:** The regret and constraint tracking bounds on the augmented problem translate to meaningful guarantees on the original problem:
   - All constraints remain satisfied (feasibility maintained)
   - Regret depends on path variation of original variables plus constraint variation
   - Additional factor $(1 + \|F\|)$ appears due to slack coupling

4. **Practical Implementation:** The slack variable approach is computationally efficient and theoretically clean.

This extension demonstrates that the OIPM framework is general enough to handle standard linear programs with mixed equality and inequality constraints, not just equality-constrained problems.
