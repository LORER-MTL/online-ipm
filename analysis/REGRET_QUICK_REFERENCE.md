# Quick Reference: Regret Bound Derivation

## The 8-Step Journey from Original Problem to Final Bound

### Starting Point
**Original Problem:** 
```
minimize    c^T x
subject to  Ax = b_t,  Fx ≤ g_t
```

**Goal:** Bound the dynamic regret $R_d(T) = \sum_{t=1}^T [c^T x_t - c^T x_t^*]$

---

## Step 1: Define Regret
$$R_d(T) = \sum_{t=1}^T [c^T x_t - c^T x_t^*]$$
- $x_t$ = algorithm's solution at time $t$
- $x_t^*$ = optimal solution at time $t$

---

## Step 2: Slack Variable Transformation
Introduce $s_i = g_{t,i} - (Fx)_i$ for each inequality.

**Augmented problem:**
$$\text{minimize } \tilde{c}^T z \quad \text{s.t. } \tilde{A}z = \tilde{b}_t, \quad z_{n+1:n+p} > 0$$

where:
$$z = \begin{bmatrix} x \\ s \end{bmatrix}, \quad \tilde{c} = \begin{bmatrix} c \\ 0 \end{bmatrix}, \quad \tilde{A} = \begin{bmatrix} A & 0 \\ F & I \end{bmatrix}, \quad \tilde{b}_t = \begin{bmatrix} b_t \\ g_t \end{bmatrix}$$

---

## Step 3: Regret Equivalence ⭐
**Key Insight:** Cost doesn't depend on slacks!

$$\tilde{c}^T z = c^T x + 0^T s = c^T x$$

Therefore:
$$\boxed{R_d^{\text{aug}}(T) = R_d(T)}$$

This means bounding augmented regret is **equivalent** to bounding original regret!

---

## Step 4: Apply Theorem 1
**Theorem 1** (from paper) for augmented problem:
$$R_d^{\text{aug}}(T) \leq \frac{11\nu_f\beta}{5\eta_0(\beta-1)} + \|\tilde{c}\| \cdot V_T^{\text{aug}}$$

For our problem:
- Barrier complexity: $\nu_f = p$ (number of inequalities)
- Cost norm: $\|\tilde{c}\| = \|c\|$
- Augmented path variation: $V_T^{\text{aug}} = \sum_{t=1}^T \|z_t^* - z_{t-1}^*\|$

Substituting:
$$R_d^{\text{aug}}(T) \leq \frac{11p\beta}{5\eta_0(\beta-1)} + \|c\| \cdot V_T^{\text{aug}}$$

---

## Step 5: Decompose Augmented Path Variation
Use triangle inequality:
$$\|z_t^* - z_{t-1}^*\| = \left\|\begin{bmatrix} x_t^* - x_{t-1}^* \\ s_t^* - s_{t-1}^* \end{bmatrix}\right\| \leq \|x_t^* - x_{t-1}^*\| + \|s_t^* - s_{t-1}^*\|$$

Sum over time:
$$\boxed{V_T^{\text{aug}} \leq V_T^x + V_T^s}$$

where:
- $V_T^x = \sum_{t=1}^T \|x_t^* - x_{t-1}^*\|$ (path variation of $x$)
- $V_T^s = \sum_{t=1}^T \|s_t^* - s_{t-1}^*\|$ (path variation of slacks)
- $V_g^{\text{ineq}} = \sum_{t=1}^T \|g_t - g_{t-1}\|$ (inequality RHS variation)

---

## Step 6: Slack Coupling Analysis
**Slack formula:** $s_t^* = g_t - Fx_t^*$

**Change in slack:**
$$s_t^* - s_{t-1}^* = (g_t - g_{t-1}) - F(x_t^* - x_{t-1}^*)$$

**Apply triangle inequality:**
$$\|s_t^* - s_{t-1}^*\| \leq \|g_t - g_{t-1}\| + \|F(x_t^* - x_{t-1}^*)\|$$

**Apply submultiplicativity:** $\|Fv\| \leq \|F\| \cdot \|v\|$
$$\boxed{\|s_t^* - s_{t-1}^*\| \leq \|g_t - g_{t-1}\| + \|F\| \cdot \|x_t^* - x_{t-1}^*\|}$$

---

## Step 7: Sum Slack Bound Over Time
$$\sum_{t=1}^T \|s_t^* - s_{t-1}^*\| \leq \sum_{t=1}^T \|g_t - g_{t-1}\| + \|F\| \sum_{t=1}^T \|x_t^* - x_{t-1}^*\|$$

Define $V_g^{\text{ineq}} = \sum_{t=1}^T \|g_t - g_{t-1}\|$ (inequality constraint variation).

Then:
$$\boxed{V_T^s \leq V_g^{\text{ineq}} + \|F\| \cdot V_T^x}$$

---

## Step 8: Combine Everything
From Step 5: $V_T^{\text{aug}} \leq V_T^x + V_T^s$

From Step 7: $V_T^s \leq V_g^{\text{ineq}} + \|F\| \cdot V_T^x$

**Substitute:**
$$V_T^{\text{aug}} \leq V_T^x + (V_g^{\text{ineq}} + \|F\| \cdot V_T^x)$$
$$= (1 + \|F\|) \cdot V_T^x + V_g^{\text{ineq}}$$

**Plug into Step 4:**
$$R_d^{\text{aug}}(T) \leq \frac{11p\beta}{5\eta_0(\beta-1)} + \|c\| \cdot [(1 + \|F\|) \cdot V_T^x + V_g^{\text{ineq}}]$$

**Use Step 3:** $R_d^{\text{aug}}(T) = R_d(T)$

---

## 🎯 FINAL RESULT

$$\boxed{R_d(T) \leq \frac{11p\beta}{5\eta_0(\beta-1)} + \|c\| \left[(1 + \|F\|) \cdot V_T^x + V_g^{\text{ineq}}\right]}$$

---

## What Each Term Means

| Term | Meaning | Depends On |
|------|---------|------------|
| $\frac{11p\beta}{5\eta_0(\beta-1)}$ | **Initialization cost** | Problem structure (# inequalities) |
| $\|c\| \cdot V_T^x$ | **Direct tracking cost** | Original variable changes |
| $\|c\| \cdot \|F\| \cdot V_T^x$ | **Indirect coupling cost** | Slack follows x changes |
| $\|c\| \cdot V_g^{\text{ineq}}$ | **Constraint change cost** | Inequality RHS changes |

### Path Variation Definitions

| Symbol | Definition | Meaning |
|--------|-----------|---------|
| $V_T^x$ | $\sum_{t=1}^T \|x_t^* - x_{t-1}^*\|$ | How much optimal $x$ changes |
| $V_T^s$ | $\sum_{t=1}^T \|s_t^* - s_{t-1}^*\|$ | How much optimal slacks change |
| $V_g^{\text{ineq}}$ | $\sum_{t=1}^T \|g_t - g_{t-1}\|$ | How much inequality RHS $g_t$ changes |

Convention: Set $x_0^* = x_1^*$, $s_0^* = s_1^*$, and $g_0 = g_1$ for $t=1$ term.

---

## Key Mathematical Tools Used

1. **Triangle inequality** (Steps 5, 6, 7)
   - $\|a + b\| \leq \|a\| + \|b\|$

2. **Submultiplicativity of matrix norms** (Step 6)
   - $\|Mv\| \leq \|M\| \cdot \|v\|$

3. **Cost equivalence** (Step 3)
   - Augmented cost = Original cost (slacks have zero cost)

4. **Theorem 1 from paper** (Step 4)
   - Applies directly to augmented problem

---

## Why $(1 + \|F\|)$ Factor?

The coupling factor $(1 + \|F\|)$ arises because:

1. When $x$ changes by $\Delta x$, slacks change by $\Delta s = -F \Delta x$
2. Total movement in augmented space: 
   $$\|\Delta z\| = \|[\Delta x; \Delta s]\| \leq \|\Delta x\| + \|\Delta s\|$$
3. Bound $\|\Delta s\|$:
   $$\|\Delta s\| \leq \|\Delta g\| + \|F\| \|\Delta x\|$$
4. Combine:
   $$\|\Delta z\| \leq \|\Delta x\| + \|\Delta g\| + \|F\| \|\Delta x\| = (1 + \|F\|)\|\Delta x\| + \|\Delta g\|$$

**This factor is tight** - you cannot remove it in general!

---

## Numerical Example

With $\|c\| = 1.75$, $\|F\| = 2.67$, $V_T^x = 1.06$, $V_g^{\text{ineq}} = 2.73$, $p=3$, $\beta=1.1$, $\eta_0=1.0$:

**Constant term:**
$$\frac{11 \times 3 \times 1.1}{5 \times 1.0 \times 0.1} = 72.6$$

**Path term:**
$$1.75 \times [(1 + 2.67) \times 1.06 + 2.73] = 1.75 \times 6.64 = 11.6$$

**Total bound:**
$$R_d(T) \leq 72.6 + 11.6 = 84.2$$

---

## Visual Aids

See these diagrams for visual explanation:
- `regret_derivation_flowchart.png` - Complete 8-step flow
- `regret_coupling_diagram.png` - Why $(1 + \|F\|)$ appears
- `regret_bound_breakdown.png` - Numerical breakdown

---

## Common Questions

**Q: Why not just bound $V_T^{\text{aug}}$ directly?**
A: We could, but decomposing it into $V_T^x$ and $V_T^s$ gives interpretable bounds in terms of the *original* problem's dynamics.

**Q: Is the $(1 + \|F\|)$ factor tight?**
A: Yes! Consider $x$ changing with $g$ fixed: slacks must change by exactly $-F\Delta x$, giving the full $\|F\|$ factor.

**Q: What if $g_t$ is constant over time?**
A: Then $V_g^{\text{ineq}} = 0$ and the bound simplifies to:
$$R_d(T) \leq \frac{11p\beta}{5\eta_0(\beta-1)} + \|c\|(1 + \|F\|) \cdot V_T^x$$

**Q: Can we reduce the constant term $O(p)$?**
A: Not easily - it comes from initializing the barrier method. Better initialization might reduce the constant, but $O(p)$ dependence is inherent.

---

## Summary

The derivation chains together:
1. Problem transformation (slacks)
2. Cost equivalence (key insight!)
3. Direct application of Theorem 1
4. Careful decomposition and bounding

The result is a **tight, interpretable bound** that captures both initialization costs and tracking costs in terms of the original problem's structure and dynamics.
