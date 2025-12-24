# Clarification: Precise Definition of $V_g^{\text{ineq}}$

## Issue Identified

The original documentation used $V_g^{\text{ineq}}$ without providing a precise mathematical definition, only describing it intuitively as "inequality RHS changes."

## Resolution

**Precise Definition:**

$$\boxed{V_g^{\text{ineq}} = \sum_{t=1}^T \|g_t - g_{t-1}\|}$$

where:
- $g_t \in \mathbb{R}^p$ is the inequality constraint RHS vector at time $t$
- $\|\cdot\|$ denotes the Euclidean (2-norm)
- Convention: $g_0 = g_1$ so the $t=1$ term is zero

## What It Measures

$V_g^{\text{ineq}}$ is the **total variation** of the inequality constraint right-hand side over the time horizon $[1, T]$.

### Physical Interpretation

If you think of $g_t$ as a point in $\mathbb{R}^p$:
- $V_g^{\text{ineq}}$ is the total distance traveled by this point from time 1 to time $T$
- It measures **how much the environment changes** the constraints
- It is **exogenous** - determined by external factors, not by algorithm choices

### Examples

1. **Static constraints:** If $g_t = g_1$ for all $t$, then $V_g^{\text{ineq}} = 0$

2. **Linear drift:** If $g_t = g_1 + (t-1) \delta$ for some $\delta \in \mathbb{R}^p$:
   $$V_g^{\text{ineq}} = \sum_{t=2}^T \|\delta\| = (T-1) \|\delta\|$$

3. **Random walk:** If $g_t = g_{t-1} + \epsilon_t$ where $\epsilon_t$ are random perturbations:
   $$V_g^{\text{ineq}} = \sum_{t=2}^T \|\epsilon_t\|$$

## Where It Appears

### In the Slack Coupling Inequality

$$\|s_t^* - s_{t-1}^*\| \leq \underbrace{\|g_t - g_{t-1}\|}_{\text{environment}} + \underbrace{\|F\| \cdot \|x_t^* - x_{t-1}^*\|}_{\text{solution tracking}}$$

Summing over time:
$$V_T^s \leq V_g^{\text{ineq}} + \|F\| \cdot V_T^x$$

**Key Insight:** The slack variation has two sources:
1. Direct changes in $g_t$ → contributes $V_g^{\text{ineq}}$
2. Induced changes from $x_t^*$ → contributes $\|F\| \cdot V_T^x$

### In the Final Regret Bound

$$R_d(T) \leq \frac{11p\beta}{5\eta_0(\beta-1)} + \|c\| \left[\underbrace{(1 + \|F\|) V_T^x}_{\text{solution tracking}} + \underbrace{V_g^{\text{ineq}}}_{\text{environment changes}}\right]$$

**Structure:**
- **Constant term:** Barrier initialization cost
- **Solution tracking:** $(1 + \|F\|) V_T^x$ captures how algorithm tracks optimal $x$ (with coupling)
- **Environment tracking:** $V_g^{\text{ineq}}$ captures how constraints themselves change

## Relationship to Other Quantities

| Quantity | Type | What It Measures |
|----------|------|------------------|
| $V_g^{\text{ineq}}$ | **Environment** | How much constraint RHS changes |
| $V_T^x$ | **Solution** | How much optimal $x$ changes |
| $V_T^s$ | **Derived** | How much optimal slacks change |

**Coupling relation:**
$$V_T^s \leq V_g^{\text{ineq}} + \|F\| \cdot V_T^x$$

This shows $V_T^s$ depends on **both** environment changes and solution changes.

## Why It's Not $V_g = \|g_T - g_1\|$

A common mistake is to think:
$$V_g^{\text{ineq}} \stackrel{?}{=} \|g_T - g_1\| \quad \text{(WRONG!)}$$

This is the **displacement** (endpoint distance), not the **path variation**.

**Correct definition uses the sum:**
$$V_g^{\text{ineq}} = \sum_{t=1}^T \|g_t - g_{t-1}\|$$

**Example showing the difference:**
- Path: $g_1 = [0, 0] \to g_2 = [1, 0] \to g_3 = [0, 0]$
- Displacement: $\|g_3 - g_1\| = 0$
- Path variation: $\|g_2 - g_1\| + \|g_3 - g_2\| = 1 + 1 = 2$

The path variation captures **all the changes**, not just net displacement.

## Numerical Example

### Problem Setup
- $p = 2$ inequalities
- $T = 4$ time steps
- Constraint RHS sequence:

| $t$ | $g_t$ |
|-----|-------|
| 0 | $[10.0, 5.0]$ |
| 1 | $[10.0, 5.0]$ |
| 2 | $[10.5, 5.2]$ |
| 3 | $[10.3, 5.4]$ |
| 4 | $[10.6, 5.1]$ |

### Calculation

$$\begin{align}
t=1: \quad &\|g_1 - g_0\| = \|[0, 0]\| = 0 \\
t=2: \quad &\|g_2 - g_1\| = \|[0.5, 0.2]\| = \sqrt{0.25 + 0.04} = 0.539 \\
t=3: \quad &\|g_3 - g_2\| = \|[-0.2, 0.2]\| = \sqrt{0.04 + 0.04} = 0.283 \\
t=4: \quad &\|g_4 - g_3\| = \|[0.3, -0.3]\| = \sqrt{0.09 + 0.09} = 0.424
\end{align}$$

$$\boxed{V_g^{\text{ineq}} = 0 + 0.539 + 0.283 + 0.424 = 1.246}$$

### Impact on Regret

If $\|c\| = 1.5$, this contributes:
$$\|c\| \cdot V_g^{\text{ineq}} = 1.5 \times 1.246 = 1.869$$
to the regret bound.

## Implementation in Code

```python
def compute_V_g_inequality(g_sequence: List[np.ndarray]) -> float:
    """
    Compute inequality constraint variation V_g^{ineq}.
    
    Args:
        g_sequence: List of g_t vectors, length T
        
    Returns:
        V_g_inequality: Sum of ||g_t - g_{t-1}|| over t=1,...,T
    """
    T = len(g_sequence)
    V_g = 0.0
    
    for t in range(1, T):
        delta_g = g_sequence[t] - g_sequence[t-1]
        V_g += np.linalg.norm(delta_g)
    
    return V_g
```

**Note:** We start loop at `t=1` (not `t=0`) because the $t=1$ term is zero by convention ($g_0 = g_1$).

## Updated Documentation

The following files now have the precise definition:

1. **`NOTATION_REFERENCE.md`** - Complete notation guide with examples
2. **`REGRET_DERIVATION_DETAILED.md`** - Added to notation section
3. **`REGRET_QUICK_REFERENCE.md`** - Added to path variation table
4. **`inequality_extension.md`** - Clarified in Section 3.1
5. **`REGRET_DOCUMENTATION_INDEX.md`** - Updated glossary

### New Visual Aid

**`vg_ineq_definition.png`** - Diagram showing:
- Time series of $g_t$ changes
- How environment and solution changes couple
- Final regret bound structure

## Key Takeaways

✅ **Definition:** $V_g^{\text{ineq}} = \sum_{t=1}^T \|g_t - g_{t-1}\|$

✅ **Type:** Exogenous environment changes (not algorithm-dependent)

✅ **Units:** Same as $g$ (e.g., if $g$ is in meters, $V_g^{\text{ineq}}$ is in meters)

✅ **Appears in:** Slack coupling inequality and final regret bound

✅ **Distinct from:** 
- $V_T^x$ (solution changes)
- $V_T^s$ (slack changes)  
- $\|g_T - g_1\|$ (displacement, not path variation)

✅ **Convention:** Set $g_0 = g_1$ so first term is zero

---

**Summary:** The issue has been fully resolved with precise mathematical definitions, numerical examples, visual aids, and updated documentation across all files.
