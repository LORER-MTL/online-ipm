# Systematic Verification of All Lemmas and Theorems

## Overview

After finding critical errors in Issues 1.1-1.3, we need to systematically check **every** mathematical statement in the paper. This document goes through each lemma, theorem, and remark to identify additional errors.

---

## LEMMA 1: Projection Formula

### Statement (Equation 2)

$$x_t' = \underset{x}{\arg\min} \|x - x_{t-1}\| \quad \text{s.t.} \quad A_t x = b_t$$

**Solution given:**
$$x_t' = x_{t-1} + A_t^\top (A_t A_t^\top)^{-1}(b_t - A_t x_{t-1})$$

### Verification

This is the **standard projection formula** onto an affine subspace. Let me verify:

**Setup:** Minimize $\frac{1}{2}\|x - x_{t-1}\|^2$ subject to $A_t x = b_t$.

**Lagrangian:**
$$L(x, \lambda) = \frac{1}{2}\|x - x_{t-1}\|^2 + \lambda^\top(A_t x - b_t)$$

**KKT conditions:**
$$\nabla_x L = x - x_{t-1} + A_t^\top \lambda = 0 \implies x = x_{t-1} - A_t^\top \lambda$$
$$A_t x = b_t$$

**Substitute:**
$$A_t(x_{t-1} - A_t^\top \lambda) = b_t$$
$$A_t x_{t-1} - A_t A_t^\top \lambda = b_t$$
$$\lambda = (A_t A_t^\top)^{-1}(A_t x_{t-1} - b_t)$$

**Therefore:**
$$x = x_{t-1} - A_t^\top (A_t A_t^\top)^{-1}(A_t x_{t-1} - b_t)$$
$$x = x_{t-1} + A_t^\top (A_t A_t^\top)^{-1}(b_t - A_t x_{t-1})$$

✓ **CORRECT**

**Note:** Requires $A_t$ to have full row rank ($\text{rank}(A_t) = p < n$) for $(A_t A_t^\top)^{-1}$ to exist. They state this assumption.

---

## LEMMA 2: Reduced Problem Hessian

### Statement

> The Hessian of the reduced problem is:
> $$\nabla^2 \tilde{f}_t(z^*) = F_t \nabla^2 f_t(x_t^*) F_t^\top$$

### Issues Already Identified

**Issue 1.2:** Wrong formula (should be $F_t^\top \nabla^2 f_t F_t$)

**Issue 1.3:** Nonsensical inverse formula

### Additional Check: Are the Bounds at Least Correct?

**Their claim:**
$$h I \preceq \nabla^2 \tilde{f}_t(z) \preceq H I$$

**Question:** Even with the wrong formula, are these bounds valid?

**Answer:** **YES, by accident!**

Since $F_t^\top F_t = I$ (orthonormal columns), we have $\|F_t\| = 1$.

For any symmetric matrix $H$:
- $\|F^\top H F\| \leq \|F^\top\| \|H\| \|F\| = \|H\|$
- If $H \succeq h I$, then $F^\top H F \succeq h F^\top F = h I$

So the bounds are correct **despite** the wrong intermediate formula.

**Verdict:** Formula wrong, bounds accidentally correct. ✓ (with caveat)

---

## LEMMA 3: Local Convergence

### Statement (Equation 9)

**Stated:**
$$\|x_{t+1} - x_t^*\| \leq \frac{2L}{h}\|x_t - x_t^*\|$$

### Issue 1.4: Missing Square

This should be **quadratic convergence** (standard Newton's method):
$$\|x_{t+1} - x_t^*\| \leq \frac{2L}{h}\|x_t - x_t^*\|^2$$

### Verification in Their Own Proof

Let me check their derivation (equations 10-12):

**Equation (10):** (Taylor expansion)
$$\nabla f_t(x_t) = \nabla f_t(x_t^*) + \int_0^1 \nabla^2 f_t(x_t^* + s(x_t - x_t^*))(x_t - x_t^*) ds$$

At optimum, $\nabla f_t(x_t^*) + A_t^\top \lambda_t^* = 0$, so:
$$\nabla f_t(x_t) = -A_t^\top \lambda_t^* + \nabla^2 f_t(\xi_t)(x_t - x_t^*)$$

**Equation (11):** (Projected gradient)
$$F_t^\top \nabla f_t(x_t) = F_t^\top \nabla^2 f_t(\xi_t)(x_t - x_t^*)$$

**Equation (12):** (Newton step)
$$\|x_{t+1} - x_t^*\| \leq \|x_t - x_t^*\| + \|[\nabla^2 f_t(x_t)]^{-1} F_t F_t^\top \nabla^2 f_t(\xi_t)(x_t - x_t^*)\|$$

They then claim:
$$\leq \left(1 + \frac{2L}{h}\right)\|x_t - x_t^*\|$$

### The Error

Their equation (12) already has the structure:
$$\|x_{t+1} - x_t^*\| \sim \|\text{inverse Hessian}\| \cdot \|\text{Hessian variation}\| \cdot \|x_t - x_t^*\|$$

For Newton's method, the **gradient** at $x_t$ is $O(\|x_t - x_t^*\|)$ (linear), and the Newton step divides by the Hessian, giving:
$$\|x_{t+1} - x_t^*\| \sim \frac{\|\nabla f_t(x_t)\|}{\|\nabla^2 f_t(x_t)\|} \sim \|x_t - x_t^*\|$$

But they're using a **quadratic** approximation of the gradient error via:
$$\nabla f_t(x_t) - \nabla f_t(x_t^*) \approx \nabla^2 f_t(x_t^*)(x_t - x_t^*)$$

The error in this approximation is:
$$O(\|x_t - x_t^*\|^2)$$

So the correct bound is:
$$\boxed{\|x_{t+1} - x_t^*\| \leq \frac{2L}{h}\|x_t - x_t^*\|^2}$$

**Verdict:** ✗ **ERROR** - Missing square, destroying quadratic convergence claim.

**Impact:** This is actually **severe** because:
- Linear convergence ($\|e_{k+1}\| \leq C\|e_k\|$) requires $C < 1$ for convergence
- Their constant $C = 2L/h$ could be $> 1$, meaning **divergence**!
- Quadratic convergence ($\|e_{k+1}\| \leq C\|e_k\|^2$) works for any $C$ if $\|e_0\|$ is small enough

---

## LEMMA 4: Quadratic Upper Bound

### Statement (Equation 13)

$$f_t(x) \leq f_t(x_t^*) + \nabla f_t(x_t^*)^\top(x - x_t^*) + \frac{H}{2}\|x - x_t^*\|^2$$

### Verification

This is the standard bound for $H$-smooth functions (Lipschitz Hessian with constant $H$).

**From Nesterov (Lemma 1.2.4):** If $\|\nabla^2 f(x)\| \leq H$, then:
$$|f(y) - f(x) - \nabla f(x)^\top(y - x)| \leq \frac{H}{2}\|y - x\|^2$$

At $x = x_t^*$ with $\nabla f_t(x_t^*) + A_t^\top \lambda_t^* = 0$:
$$f_t(x) \leq f_t(x_t^*) + \nabla f_t(x_t^*)^\top(x - x_t^*) + \frac{H}{2}\|x - x_t^*\|^2$$

✓ **CORRECT** (assuming Assumption 1: Lipschitz Hessian)

---

## THEOREM 1: Dynamic Regret Bound

### Statement

Under Conditions 1-4:
$$R_T \leq V_T + 1$$

### Conditions

**Condition 1:** $\|x_t' - x_t^*\| \leq v$ (constraint variation bounded)

**Condition 2:** $v \leq \gamma - \frac{2L}{h}\gamma^2$ (variation small enough for convergence)

**Condition 3:** $\|x_0 - x_0^*\| \leq \gamma$ (good initialization)

**Condition 4:** $\gamma \leq \beta$ (within local region)

### Issue 1.6: Condition 2 Forces $v \approx 0$

Let me re-examine this more carefully.

**Condition 2:** $v \leq \gamma - \frac{2L}{h}\gamma^2$

They then choose $\gamma = \frac{h}{2L}$ to maximize the right-hand side.

**Compute:**
$$\gamma - \frac{2L}{h}\gamma^2 = \frac{h}{2L} - \frac{2L}{h} \cdot \frac{h^2}{4L^2} = \frac{h}{2L} - \frac{h}{2L} = 0$$

So: $v \leq 0$, which means $v = 0$ (since $v \geq 0$ by definition of norm).

**What does $v = 0$ mean?**

$$v = \|x_t' - x_t^*\| = 0 \implies x_t' = x_t^*$$

This means: **after projecting onto the new constraint, you're already at the optimum!**

This is only possible if:
1. $x_{t-1}^* = x_t^*$ (no variation in optima), OR
2. The projection magically lands exactly at the new optimum

### Deeper Analysis

Wait, let me reconsider. Maybe I misread their proof.

Looking at their proof of Theorem 1:
- They use Lemma 3 to bound $\|x_{t+1} - x_t^*\|$ in terms of $\|x_t - x_t^*\|$
- They need $\|x_t - x_t^*\| \leq \gamma$ to maintain the local convergence region

**After projection:**
$$\|x_t' - x_t^*\| = v$$

**After Newton step:**
$$\|x_{t+1} - x_t^*\| \leq \frac{2L}{h}\|x_t' - x_t^*\| = \frac{2L}{h}v$$

(Using their wrong linear convergence formula from Lemma 3)

**To stay in the local region:**
$$\|x_{t+1} - x_t^*\| \leq \gamma$$
$$\frac{2L}{h}v \leq \gamma$$
$$v \leq \frac{h}{2L}\gamma$$

Hmm, this gives $v \leq \frac{h\gamma}{2L}$, not what they wrote.

Let me check their Condition 2 again...

**Their Condition 2:**
$$v \leq \gamma - \frac{2L}{h}\gamma^2$$

This looks like they're using:
$$\|x_{t+1} - x_t^*\| \leq \|x_t' - x_t^*\| + \text{Newton correction}$$

If the Newton correction is $\frac{2L}{h}\gamma^2$ and we need $\|x_{t+1} - x_t^*\| \leq \gamma$:
$$v + \frac{2L}{h}\gamma^2 \leq \gamma$$
$$v \leq \gamma - \frac{2L}{h}\gamma^2$$

**But this assumes the correction is $O(\gamma^2)$, which would be true for quadratic convergence, not their linear formula!**

### The Contradiction

- **Lemma 3** says: $\|x_{t+1} - x_t^*\| \leq \frac{2L}{h}\|x_t - x_t^*\|$ (linear)
- **Condition 2** assumes: Newton correction is $O(\gamma^2)$ (quadratic)

These are **inconsistent**!

**If we use the correct quadratic convergence:**
$$\|x_{t+1} - x_t^*\| \leq \frac{2L}{h}\|x_t' - x_t^*\|^2 = \frac{2L}{h}v^2$$

**To stay in local region:**
$$\frac{2L}{h}v^2 \leq \gamma$$
$$v^2 \leq \frac{h\gamma}{2L}$$
$$v \leq \sqrt{\frac{h\gamma}{2L}}$$

**If we choose $\gamma = \frac{h}{2L}$:**
$$v \leq \sqrt{\frac{h}{2L} \cdot \frac{h}{2L}} = \sqrt{\frac{h^2}{4L^2}} = \frac{h}{2L} = \gamma$$

So we'd need $v \leq \gamma$, which is **much more reasonable**!

**Verdict:** ✗ **ERROR** - Condition 2 is based on quadratic convergence, but Lemma 3 states linear convergence. These are inconsistent. With the correct quadratic formula, Condition 2 would allow $v = O(\sqrt{\gamma})$ instead of $v = 0$.

---

## THEOREM 2: Path Length Bound

### Statement

$$V_T = \sum_{t=1}^T \|x_t^* - x_{t-1}^*\| \leq \sum_{t=1}^T \|x_t^* - x_t'\| + \|x_t' - x_{t-1}^*\|$$

Then uses:
$$\|x_t' - x_{t-1}^*\| \leq \|x_t' - x_{t-1}\| + \|x_{t-1} - x_{t-1}^*\|$$

### Verification

This is just the **triangle inequality**, repeated twice:
$$\|x_t^* - x_{t-1}^*\| \leq \|x_t^* - x_t'\| + \|x_t' - x_{t-1}^*\|$$
$$\|x_t' - x_{t-1}^*\| \leq \|x_t' - x_{t-1}\| + \|x_{t-1} - x_{t-1}^*\|$$

✓ **CORRECT** (trivial algebra)

### Issue: Circular Reasoning?

The bound becomes:
$$V_T \leq \sum_{t=1}^T (v + \|x_t' - x_{t-1}\| + \|x_{t-1} - x_{t-1}^*\|)$$

They then bound:
- $\|x_{t-1} - x_{t-1}^*\| \leq \gamma$ (by induction, assuming Conditions hold)
- $\|x_t' - x_{t-1}\|$ using Lemma 1

**But to maintain $\|x_t - x_t^*\| \leq \gamma$ for all $t$, they need Condition 2, which requires $v = 0$!**

So the theorem is only non-trivial if $v > 0$, but $v > 0$ violates Condition 2!

**Verdict:** ⚠ **CIRCULAR** - Theorem requires conditions that make the result trivial.

---

## ASSUMPTION VERIFICATION

### Assumption 1: Lipschitz Continuous Hessian

**Statement:** $\|\nabla^2 f_t(x) - \nabla^2 f_t(y)\| \leq L\|x - y\|$ for all $x, y$ with $\|x - x_t^*\|, \|y - x_t^*\| \leq \beta$

This is a **standard assumption** for Newton's method analysis.

✓ **CORRECT** (standard)

### Assumption 2: Bounded Hessian Inverse

**Statement:** $\|\nabla^2 f_t(x)^{-1}\| \leq 1/h$ for all $x$ with $\|x - x_t^*\| \leq \beta$

**Issue:** This is **strong convexity** in a ball, which is reasonable.

However, **their numerical example violates this!**

For $f(x) = \alpha e^{\beta|x|}$:
- At $x \neq 0$: $\nabla^2 f = \alpha\beta^2 e^{\beta|x|}$ (exists)
- At $x = 0$: Hessian **does not exist** (non-differentiable)

So the numerical example violates their own assumption.

✓ **CORRECT** (as an assumption)  
✗ **VIOLATED** (in their experiments)

### Assumption 3: Lipschitz Function Values

**Statement:** $\|f_t(x) - f_t(x_t^*)\| \leq l\|x - x_t^*\|$

**Issues:**

1. **Notation:** Should be $|f_t(x) - f_t(x_t^*)|$ (absolute value for scalars)

2. **Non-standard:** Usually assume Lipschitz **gradient**, not function values. Lipschitz gradient implies Lipschitz function, but not vice versa.

3. **Not used in proofs?** I don't see where they actually use this assumption.

⚠ **SUSPICIOUS** - Non-standard, possibly unnecessary

---

## REMARK 1: Initialization

**Statement:** "Finding an initial point $x_0$ satisfying Condition 3 can be done by solving (1) to high accuracy at time $t = 0$."

**Issue:** This undermines the "online" nature:
- You need to solve the first problem **offline** to high accuracy
- If you can do this once, why not do it every time?
- Defeats the purpose of an online algorithm

⚠ **CONCEPTUAL ISSUE** - Not an error per se, but weakens the claims

---

## REMARK 2: Complexity

**Statement:** "The time-complexity of OEN-M and OPEN-M are dominated by the matrix inversion step which is $O(n^5 \log(n))$ in the general case."

### Issue 1.1: Completely Wrong

- Matrix inversion: $O(n^3)$ (Gaussian elimination)
- You should never invert matrices (solve $Hx = b$ instead)
- Specialized methods (Cholesky, CG) can be faster

✗ **CRITICAL ERROR** - Off by $O(n^2 \log n)$ factor!

### Missing: Null Space Computation

Computing $F_t$ (null space basis) requires:
- QR decomposition of $A_t^\top$: $O(np^2)$
- SVD: $O(np^2)$ or $O(n^2 p)$

For time-varying $A_t$, this is done **every iteration** and is expensive!

✗ **INCOMPLETE ANALYSIS**

---

## REMARK 3: Comparison to Prior Work

**Statement:** "OPEN-M possesses the tightest dynamic regret bounds of any previously proposed online equality-constrained algorithm. The method is also parameter-free and computationally efficient."

### Issue 3.2: Unsubstantiated "Tightest Bounds"

They claim $O(V_T + 1)$ is tightest, but:
- References [6], [14], [15] also get $O(V_T)$
- Constant factors are not compared
- No lower bounds provided

✗ **FALSE CLAIM** - Not proven

### Issue 3.3: Not Parameter-Free

The algorithm requires:
- $h$, $L$, $\beta$ (in Condition 2, 4)
- Initial point with $\|x_0 - x_0^*\| \leq \gamma$

These are **problem-dependent parameters**.

✗ **MISLEADING**

---

## SECTION IV NUMERICAL EXAMPLE

### Issue 4.1: Non-Differentiable Cost

**Used:** $f_i(x) = \alpha_i e^{\beta_i|x|}$

**Problems:**
- $|x|$ not differentiable at $x = 0$
- Violates Assumptions 1, 2
- Newton's method cannot execute at $x = 0$

✗ **CRITICAL ERROR** - Experiments violate assumptions

### Issue 4.2: Diagonal Hessian Still Time-Varying

**Claim:** "The fixed nature of the network and the diagonal Hessian matrix means that the inversion step only has to be done once."

**Reality:**
- $\nabla^2 f_t = \text{diag}(\alpha_1 \beta_1^2 e^{\beta_1|x_1|}, \ldots)$ changes with $t$ because $\alpha_i, \beta_i$ are resampled
- **Must refactor every iteration**

✗ **FALSE CLAIM**

### Issue 4.3: Network Topology Inconsistent

**Claim:** "15 nodes connected via 30 arcs" in a "radial (tree) network"

**Problem:** Tree with 15 nodes has 14 edges (28 directed arcs)

✗ **INCONSISTENT**

### Issue 4.4: Unfair Comparison

Comparing:
- OPEN-M: equality constraints $Ax = b$
- MOSP, MALM: inequality constraints $Ax \leq b$

**These are different problems!** Inequality constraints have a larger feasible set.

✗ **INVALID COMPARISON**

### Issue 4.5: No Reproducibility

Missing:
- Code
- Exact parameters ($\alpha_i, \beta_i$ distributions)
- Initial conditions
- Random seeds

✗ **NOT REPRODUCIBLE**

---

## NEW ISSUES DISCOVERED

### Issue 1.7: Lemma 3 Contradicts Theorem 1

**Lemma 3:** Linear convergence $\|e_{t+1}\| \leq C\|e_t\|$ with $C = 2L/h$

**Problem:** If $C > 1$, this is **divergence**, not convergence!

For convergence, need $C < 1$, i.e., $2L/h < 1$, i.e., $L < h/2$.

**But they never assume this!**

**Theorem 1 Condition 2:** Uses quadratic correction $O(\gamma^2)$, inconsistent with Lemma 3.

✗ **NEW CRITICAL ERROR** - Lemma 3 can imply divergence, not convergence!

### Issue 1.8: No Analysis of $\|x_t' - x_{t-1}\|$

**In Theorem 2**, they claim to bound $V_T$ using:
$$\|x_t' - x_{t-1}\| = \text{distance traveled by projection}$$

**But they never actually bound this term!**

From Lemma 1:
$$x_t' - x_{t-1} = A_t^\top(A_t A_t^\top)^{-1}(b_t - A_t x_{t-1})$$

Taking norms:
$$\|x_t' - x_{t-1}\| \leq \|A_t^\top(A_t A_t^\top)^{-1}\| \|b_t - A_t x_{t-1}\|$$

This depends on:
- Condition number of $A_t A_t^\top$
- Changes in $b_t$ and $A_t$

**They never bound this!**

✗ **NEW ERROR** - Incomplete proof of Theorem 2

### Issue 1.9: "Unitary Matrix" Misuse

**Statement (Section II.B):** "there exists a unitary matrix $F_t \in \mathbb{R}^{n \times (n-p)}$"

**Problem:** 
- "Unitary" means $U U^* = U^* U = I$ (requires square matrix)
- They mean "semi-orthogonal" or "matrix with orthonormal columns"

✓ **Terminology error** (minor, but sloppy)

---

## SUMMARY OF ALL ISSUES

| # | Issue | Location | Severity | Type |
|---|-------|----------|----------|------|
| 1.1 | $O(n^5 \log n)$ complexity | Remark 2 | **CRITICAL** | Wrong analysis |
| 1.2 | Wrong Hessian formula | Lemma 2 | **HIGH** | Wrong math |
| 1.3 | Nonsensical inverse formula | Lemma 2 proof | **HIGH** | Wrong math |
| 1.4 | Missing square in convergence | Lemma 3 (Eq. 9) | **HIGH** | Wrong math |
| 1.5 | KKT invertibility condition | Section II.B | **MEDIUM** | Wrong reasoning |
| 1.6 | Condition 2 forces $v = 0$ | Theorem 1 | **CRITICAL** | Makes result trivial |
| 1.7 | **NEW:** Linear convergence $\Rightarrow$ divergence if $L > h/2$ | Lemma 3 | **CRITICAL** | Wrong convergence |
| 1.8 | **NEW:** No bound on $\|x_t' - x_{t-1}\|$ | Theorem 2 proof | **HIGH** | Incomplete proof |
| 1.9 | "Unitary" for non-square matrix | Section II.B | Low | Terminology |
| 2.1 | Non-smooth cost $\|x\|$ | Section IV | **CRITICAL** | Violates assumptions |
| 2.2 | Network topology inconsistent | Section IV | **MEDIUM** | Inconsistent data |
| 2.3 | Diagonal Hessian still time-varying | Section IV | **HIGH** | False claim |
| 2.4 | Unfair algorithm comparison | Section IV | **HIGH** | Invalid experiment |
| 2.5 | No reproducibility | Section IV | **MEDIUM** | Missing details |
| 3.1 | False "first second-order" claim | Section III.A | **CRITICAL** | False novelty |
| 3.2 | "Tightest bounds" unproven | Remark 3 | **MEDIUM** | Unsubstantiated |
| 3.3 | Not parameter-free | Remark 3 | **HIGH** | Misleading |
| 4.1 | No lower bounds | Throughout | **MEDIUM** | Incomplete |
| 4.2 | Initialization requires offline solve | Remark 1 | **MEDIUM** | Weakens claims |
| 4.3 | Missing null space cost | Throughout | **HIGH** | Incomplete analysis |
| 4.4 | Assumption 3 never used | Section II.C | Low | Unnecessary |

**Total:** 21 issues (13 Critical/High severity)

---

## OVERALL ASSESSMENT

### Can Anything Be Trusted?

**YES (with caveats):**
1. **Lemma 1** (projection formula): Correct ✓
2. **Lemma 4** (quadratic upper bound): Correct ✓
3. **Basic algorithm** (project then Newton): Sound ✓

**NO:**
1. **Lemma 2** (Hessian formula, inverse): Wrong formulas, bounds accidentally correct
2. **Lemma 3** (convergence): Missing square, can imply divergence
3. **Theorem 1** (dynamic regret): Conditions inconsistent and may force $v = 0$
4. **Theorem 2** (path length): Incomplete proof
5. **All numerical experiments**: Violate assumptions
6. **Novelty claims**: False
7. **Complexity analysis**: Completely wrong

### Bottom Line

The paper has **pervasive errors** across:
- Linear algebra (wrong formulas)
- Convergence analysis (wrong rate, potential divergence)
- Proof technique (incomplete, circular reasoning)
- Experiments (violate assumptions, unreproducible)
- Literature review (false novelty claims)

**This is not salvageable with minor corrections. It needs a complete rewrite.**

The only thing that survives is the basic algorithmic idea: "project onto new constraint, then take a Newton step in the null space." But this idea is not novel—it's standard projected Newton method applied to time-varying constraints.
