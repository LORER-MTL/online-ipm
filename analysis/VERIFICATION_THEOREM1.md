# Mathematical Verification: Theorem 1 Application

## Question: Is Theorem 1 correctly applied to the augmented problem?

Let me verify the conditions and application carefully.

---

## Theorem 1 Statement (from Paper)

**Original Statement:**

For the online optimization problem:
```
minimize    d_t(z, η) = η · c^T z + φ(z)
subject to  Az = b_t
```

where:
- $\phi(z)$ is a $\nu_f$-self-concordant barrier
- Barrier parameter $\eta_t$ is updated as $\eta_{t+1} = \beta \eta_t$ with $\beta > 1$
- Initial $\eta_0 > 0$

**The OIPM algorithm achieves:**

$$R_d(T) \leq \frac{11\nu_f \beta}{5\eta_0(\beta-1)} + \|c\| \cdot V_T$$

where $V_T = \sum_{t=1}^T \|z_t^* - z_{t-1}^*\|$.

---

## Our Augmented Problem

```
minimize    d_t(z, η) = η · ̃c^T z + φ(z)
subject to  Ãz = b̃_t
```

where:

$$z = \begin{bmatrix} x \\ s \end{bmatrix}, \quad \tilde{c} = \begin{bmatrix} c \\ 0 \end{bmatrix}, \quad \tilde{A} = \begin{bmatrix} A & 0 \\ F & I \end{bmatrix}, \quad \tilde{b}_t = \begin{bmatrix} b_t \\ g_t \end{bmatrix}$$

$$\phi(z) = -\sum_{i=1}^p \log(s_i)$$

---

## Verification Checklist

### ✅ 1. Is $\phi(z)$ Self-Concordant?

**Claim:** $\phi(z) = -\sum_{i=1}^p \log(s_i)$ is self-concordant with $\nu_f = p$.

**Verification:**

The logarithmic barrier $-\log(s)$ for $s > 0$ is the **canonical self-concordant barrier** for the domain $\{s : s > 0\}$.

For a product domain $\{(s_1, \ldots, s_p) : s_i > 0 \text{ for all } i\}$, the sum of logarithmic barriers:
$$\phi(s) = -\sum_{i=1}^p \log(s_i)$$

is self-concordant with complexity:
$$\nu_f = \sum_{i=1}^p 1 = p$$

**Reference:** Nesterov & Nemirovskii (1994), Boyd & Vandenberghe (2004) - standard result in interior point theory.

**Status: ✅ CORRECT**

---

### ✅ 2. What About the $x$ Variables?

**Question:** The barrier $\phi(z) = -\sum_{i=1}^p \log(s_i)$ only depends on $s$, not on $x$. Is this allowed?

**Answer: YES** ✅

The barrier only needs to be defined on the **constrained domain**. Since the constraints $Ax = b_t$ and $Fx + s = g_t$ determine $x$ and $s$ jointly, and we only need $s > 0$, the barrier $-\sum \log(s_i)$ is sufficient.

Alternatively, we can think of this as:
- The domain for $x$ is $\mathbb{R}^n$ (unconstrained in $x$)
- The domain for $s$ is $\mathbb{R}^p_{++}$ (positive orthant)
- Total barrier: $\phi(x, s) = 0 + (-\sum \log(s_i))$

The zero barrier for $x$ means $x$ is unconstrained (which is true - constraints are handled via equality constraints $Az = b_t$).

**Status: ✅ CORRECT**

---

### ✅ 3. Is $\|\tilde{c}\| = \|c\|$?

**Claim:** 
$$\|\tilde{c}\| = \left\|\begin{bmatrix} c \\ 0 \end{bmatrix}\right\| = \|c\|$$

**Verification:**

Using Euclidean 2-norm:
$$\|\tilde{c}\|_2^2 = \sum_{i=1}^n c_i^2 + \sum_{j=1}^p 0^2 = \|c\|_2^2$$

Therefore:
$$\|\tilde{c}\|_2 = \|c\|_2$$

**Status: ✅ CORRECT**

---

### ✅ 4. What is $V_T^{\text{aug}}$?

**Definition:**
$$V_T^{\text{aug}} = \sum_{t=1}^T \|z_t^* - z_{t-1}^*\|$$

where $z_t^* = [x_t^*; s_t^*]$ is the optimal solution to the augmented problem at time $t$.

**Status: ✅ CORRECTLY DEFINED**

---

### ✅ 5. Algorithm Assumptions

**Required by Theorem 1:**
- Barrier parameter update: $\eta_{t+1} = \beta \eta_t$ with $\beta > 1$
- Initial parameter $\eta_0 > 0$
- Algorithm follows OIPM procedure (solving barrier subproblems, tracking parameter)

**In our setting:**
All these are **exactly** satisfied - we apply OIPM to the augmented problem with the logarithmic barrier.

**Status: ✅ SATISFIED**

---

### ✅ 6. Constraints Time-Varying?

**Theorem 1 assumes:** Constraints $Az = b_t$ are time-varying (only RHS changes).

**Our augmented problem:** 
$$\tilde{A}z = \tilde{b}_t$$

where $\tilde{A}$ is **fixed** and $\tilde{b}_t = [b_t; g_t]$ is **time-varying**.

**Status: ✅ MATCHES THEOREM ASSUMPTIONS**

---

### ⚠️ 7. Critical Check: Constraint Variation Bound

**Theorem 1 also gives constraint violation bound:**
$$\text{Vio}(T) \leq V_b$$

where:
$$V_b = \sum_{t=1}^T \|\tilde{b}_t - \tilde{b}_{t-1}\|$$

**In our setting:**
$$\|\tilde{b}_t - \tilde{b}_{t-1}\| = \left\|\begin{bmatrix} b_t - b_{t-1} \\ g_t - g_{t-1} \end{bmatrix}\right\|$$

By the norm inequality we verified:
$$\left\|\begin{bmatrix} b_t - b_{t-1} \\ g_t - g_{t-1} \end{bmatrix}\right\|_2 = \sqrt{\|b_t - b_{t-1}\|^2 + \|g_t - g_{t-1}\|^2}$$

By triangle inequality (for sums, not concatenation):
$$\sqrt{a^2 + b^2} \leq a + b \quad \text{for } a, b \geq 0$$

**Wait - is this true?**

Let me check: $(a + b)^2 = a^2 + 2ab + b^2 \geq a^2 + b^2$ since $2ab \geq 0$.

Taking square roots: $a + b \geq \sqrt{a^2 + b^2}$ ✅

So:
$$\|\tilde{b}_t - \tilde{b}_{t-1}\| \leq \|b_t - b_{t-1}\| + \|g_t - g_{t-1}\|$$

Summing:
$$V_b \leq V_b^{eq} + V_g^{\text{ineq}}$$

**Status: ✅ CORRECT**

---

## Summary of Theorem 1 Application

| Requirement | Our Setting | Status |
|-------------|-------------|--------|
| Self-concordant barrier | $\phi = -\sum \log(s_i)$ | ✅ Valid ($\nu_f = p$) |
| Barrier complexity | $\nu_f = p$ | ✅ Correct |
| Cost vector norm | $\|\tilde{c}\| = \|c\|$ | ✅ Verified |
| Time-varying constraints | $\tilde{b}_t = [b_t; g_t]$ | ✅ Matches assumption |
| Algorithm procedure | OIPM with barrier updates | ✅ Exactly as specified |
| Path variation | $V_T^{\text{aug}}$ well-defined | ✅ Correct |

---

## Resulting Bound

**From Theorem 1:**
$$R_d^{\text{aug}}(T) \leq \frac{11p\beta}{5\eta_0(\beta-1)} + \|c\| \cdot V_T^{\text{aug}}$$

**This is rigorously correct** - all conditions of Theorem 1 are satisfied.

---

## Potential Concern: Is $p$ the Right Complexity?

**Question:** We have $n+p$ variables total. Should the barrier complexity be $n+p$?

**Answer: NO** ✅

The barrier complexity depends on the **domain constraints** that the barrier enforces:
- Variables $x \in \mathbb{R}^n$: No barrier needed (unconstrained)
- Variables $s \in \mathbb{R}^p_{++}$: Need barrier $-\sum \log(s_i)$ with complexity $p$

The equality constraints $Az = b_t$ are handled by the constraint $Az = b_t$ in the problem formulation, **not** by the barrier.

**Standard result:** For domain $\mathbb{R}^k \times \mathbb{R}^p_{++}$, the barrier complexity is $p$, not $k+p$.

**Status: ✅ CORRECT**

---

## Conclusion

**Theorem 1 is correctly applied to the augmented problem.**

All assumptions are satisfied:
- ✅ Self-concordant barrier with correct complexity
- ✅ Cost vector norm computed correctly  
- ✅ Time-varying constraints match framework
- ✅ Path variation well-defined
- ✅ Algorithm procedure matches specification

The resulting bound:
$$R_d^{\text{aug}}(T) \leq \frac{11p\beta}{5\eta_0(\beta-1)} + \|c\| \cdot V_T^{\text{aug}}$$

is **mathematically rigorous and correct**.

---

## Note on the Paper

The original OIPM paper (presumably) proves Theorem 1 for the general case of:
- Self-concordant barriers
- Time-varying equality constraints
- Barrier parameter updates

Our augmented problem is a **special case** of this framework, so Theorem 1 applies directly.

**No issues found.**
