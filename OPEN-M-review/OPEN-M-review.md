# Critical Review: "An Online Newton's Method for Time-varying Linear Equality Constraints"

**Authors:** Jean-Luc Lupien and Antoine Lesage-Landry  
**ArXiv:** 2212.02748v3 (July 2023)

---

## Executive Summary

This paper proposes OPEN-M, an online Newton method for optimization with time-varying linear equality constraints. While the core algorithmic idea (project then Newton step) is reasonable, the paper contains numerous errors in linear algebra, complexity analysis, proofs, and experimental design. Several novelty claims appear to be false.

---

## 1. Critical Errors

### 1.1 Complexity Claim (Remark 2)

**Claim:** "The time-complexity of OEN-M and OPEN-M are dominated by the matrix inversion step which is $O(n^5 \log(n))$ in the general case."

**Problems:**
1. Matrix inversion is $O(n^3)$, not $O(n^5 \log n)$
2. You should never explicitly invert matrices—solve linear systems instead
3. For Newton's method, you solve $H \Delta x = -g$ via Cholesky ($O(n^3/3)$) or iterative methods

**Correct complexity:** $O(n^3)$ for dense systems, potentially much faster for sparse systems.

### 1.2 Wrong Hessian Formula (Lemma 2)

**Their formula:**
$$\nabla^2 \tilde{f}_t(z) = F_t \nabla^2 f_t(x_t^*) F_t^\top$$

**Correct formula (chain rule):**
$$\nabla^2 \tilde{f}_t(z) = F_t^\top \nabla^2 f_t(F_t z + \hat{x}) F_t$$

With $F_t \in \mathbb{R}^{n \times (n-p)}$:
- Their formula gives an $n \times n$ matrix (wrong dimension)
- Correct formula gives $(n-p) \times (n-p)$ matrix ✓

The final bounds happen to be correct because $\|F_t\| = 1$, but the derivation is wrong.

### 1.3 Incorrect Inverse Formula (Lemma 2 Proof)

**Their claim:**
$$\|\nabla^2 \tilde{f}_t(z^*)^{-1}\| = \|F_t(F_t^\top F_t)^{-1} \nabla^2 f_t(x_t^*)^{-1} (F_t^\top F_t)^{-1} F_t^\top\|$$

This formula is nonsensical. For non-square $F$, $(F^\top H F)^{-1} \neq F^\top H^{-1} F$.

### 1.4 Equation (9) Missing Square (Lemma 3)

**Stated:** $\|x_{t+1} - x_t^*\| \leq \frac{2L}{h}\|x_t - x_t^*\|$ (linear convergence)

**Correct:** $\|x_{t+1} - x_t^*\| \leq \frac{2L}{h}\|x_t - x_t^*\|^2$ (quadratic convergence)

This is standard Newton convergence. Their own proof derives the squared version in equation (12), then incorrectly states the linear version.

### 1.5 KKT Invertibility Condition (Section II.B)

**Their claim:** "We assume that the Hessian is invertible for all $t$ which implies that $D_t(x)$ is also invertible."

**Correct condition:** The KKT matrix is invertible iff:
1. $A$ has full row rank, AND
2. $\nabla^2 f_t(x)$ is positive definite on $\mathcal{N}(A)$

An invertible Hessian with negative eigenvalues in $\mathcal{N}(A)$ makes the KKT matrix singular. They misread Boyd & Vandenberghe Section 10.1.1.

### 1.6 Condition 2 Forces $v \approx 0$ (Theorem 2)

**Required:** $v \leq \gamma - \frac{2L}{h}\gamma^2$

With $\gamma = h/(2L)$:
$$v \leq \frac{h}{2L} - \frac{2L}{h} \cdot \frac{h^2}{4L^2} = 0$$

This means **no variation in optima is allowed**, making the "time-varying" aspect trivial.

---

## 2. Terminology Errors

### 2.1 "Unitary Matrix" for Non-Square Matrix (Section II.B)

**Their statement:** "there exists a unitary matrix $F_t \in \mathbb{R}^{n \times (n-p)}$"

Unitary matrices must be square. They mean a matrix with **orthonormal columns** (semi-orthogonal/isometry), i.e., $F_t^\top F_t = I_{n-p}$.

### 2.2 Non-Standard Assumption 3

**Stated:** $\|f_t(x) - f_t(x_t^*)\| \leq l\|x - x_t^*\|$

Using $\|\cdot\|$ for a scalar is incorrect notation (should be $|\cdot|$). This is Lipschitz continuity of function values, not the standard Lipschitz gradient assumption.

---

## 3. False Novelty Claims

### 3.1 "First Online Second-Order Algorithm with Constraints" (Section III.A)

**Their claim:** "This is the first online, second-order algorithm that admits constraints."

**Counterexample:** Their own reference [9] (Abernethy, Hazan, Rakhlin 2012) presents interior-point methods for online learning with constraints, achieving $O(\sqrt{n} \log T)$ regret. Interior-point methods are second-order (use Hessian information via self-concordant barriers).

### 3.2 "Tightest Bounds" (Remark 3)

**Their claim:** "OPEN-M possesses the tightest dynamic regret bounds of any previously proposed online equality-constrained algorithm"

Their bound is $O(V_T + 1)$, as are bounds from [6], [14], [15]. They never compare constant factors, which could be arbitrarily large for ill-conditioned problems.

### 3.3 "Parameter-Free" (Remark 3)

**Their claim:** "The method is also parameter-free"

The algorithm requires knowing:
- $h$: bound on $\|\nabla^2 f_t(x_t^*)^{-1}\|$
- $L$: Lipschitz constant of the Hessian  
- $\beta$: radius of local Lipschitz continuity
- Initial point $x_0$ with $\|x_0 - x_0^*\| \leq \gamma$

These are problem-specific parameters hidden in assumptions.

---

## 4. Experimental Issues (Section IV)

### 4.1 Non-Smooth Cost Function

**Used:** $f_i(x) = \alpha_i e^{\beta_i |x|}$

The absolute value $|x|$ is **not differentiable at $x = 0$**, violating:
- Their claim that $f_t$ is "twice-differentiable"
- Assumption 2 (Lipschitz continuous Hessian)

### 4.2 Inconsistent Network Topology

**Claim:** "a fixed, radial network composed of 15 nodes connected via 30 arcs"

A radial (tree) network with $n$ nodes has $n-1$ edges. With 15 nodes:
- Undirected: 14 edges
- Directed (both ways): 28 arcs

30 arcs is inconsistent with a tree structure.

### 4.3 False One-Time Factorization Claim

**Claim:** "The fixed nature of the network and the diagonal Hessian matrix means that the inversion step only has to be done once."

**Reality:** The Hessian $\nabla^2 f_t$ changes with $t$ because cost parameters $\alpha_i, \beta_i$ are resampled each round. Numerical re-factorization is required every iteration; only symbolic factorization can be reused.

### 4.4 Unfair Algorithm Comparison

They compare OPEN-M (equality constraints) to MOSP and MALM (inequality constraints):

> "the equality constraint $Ax_t = b_t$ is relaxed to $Ax_t - b_t \leq 0$"

This changes the problem (larger feasible set, different optimum). MOSP and MALM solve a **different, harder problem**.

---

## 5. Missing Analysis

### 5.1 Sparsity Exploitation

The paper ignores sparsity entirely. For the network flow example:

| Operation | Dense | Sparse (tree) |
|-----------|-------|---------------|
| Factor $AA^\top$ | $O(p^3)$ | $O(p)$ |
| Newton system | $O(n^3)$ | $O(n)$ to $O(n^{1.5})$ |

**Key insight missed:** For fixed $A$ with time-varying $f_t$:
- Projection: Factor $AA^\top$ **once**, reuse forever
- Newton: Re-factor $A H_t^{-1} A^\top$ **every iteration**

The projection is actually cheaper!

### 5.2 Null Space Computation Cost

For time-varying $A_t$, computing orthonormal $F_t$ with $\mathcal{R}(F_t) = \mathcal{N}(A_t)$ requires QR/SVD decomposition: $O(np^2)$ or $O(n^2 p)$. This cost is never mentioned.

### 5.3 Constraint Qualification for Time-Varying $A_t$

No discussion of:
- What if $\text{rank}(A_t)$ changes?
- What if $\mathcal{N}(A_t) \cap \mathcal{N}(A_{t+1}) = \{0\}$?
- Sensitivity via $(A_t A_t^\top)^{-1}$ (could be ill-conditioned)

### 5.4 Non-Convex Functions: Which Optimum?

For non-convex $f_t$, there may be multiple local optima. The analysis assumes a unique, well-defined path $\{x_t^*\}$, which requires convexity or unstated assumptions.

### 5.5 Lower Bounds

They claim bounds are "tight" but provide no lower bounds showing $\Omega(V_T)$ is necessary.

### 5.6 Initialization Requirements

Finding $x_0$ with $\|x_0 - x_0^*\| \leq \gamma$ requires solving the initial problem to high accuracy offline, undermining the "online" nature.

---

## 6. Summary Table

| Issue | Section | Severity |
|-------|---------|----------|
| $O(n^5 \log n)$ complexity | Remark 2 | **Critical** |
| Wrong Hessian formula | Lemma 2 | **High** |
| Incorrect inverse formula | Lemma 2 proof | **High** |
| Equation (9) missing square | Lemma 3 | **Medium** |
| KKT invertibility condition | Section II.B | **High** |
| Condition 2 forces $v = 0$ | Theorem 2 | **Critical** |
| "Unitary" for non-square | Section II.B | Low |
| Non-standard Assumption 3 | Section II.C | Low |
| False "first second-order" claim | Section III.A | **Critical** |
| "Tightest bounds" unsubstantiated | Remark 3 | **Medium** |
| "Parameter-free" misleading | Remark 3 | **High** |
| Non-smooth cost function | Section IV | **High** |
| Inconsistent network topology | Section IV | **Medium** |
| False one-time factorization | Section IV | **High** |
| Unfair algorithm comparison | Section IV | **High** |
| Ignores sparsity | Throughout | **High** |
| Missing null space cost | Throughout | **Medium** |
| No constraint qualification | Section III.B | **Medium** |
| Non-convex ambiguity | Assumptions | **High** |
| No lower bounds | Throughout | **Medium** |
| Initialization requires offline solve | Remark 1 | **Medium** |

---

## 7. Conclusion

The paper's core algorithmic idea (projection followed by Newton step) is sound and potentially useful. However:

1. **Multiple fundamental errors** in linear algebra and complexity analysis
2. **False novelty claims** contradicted by their own references
3. **Proofs with errors** that happen to cancel (wrong intermediate steps, correct final bounds)
4. **Experimental design issues** including non-smooth objectives and unfair comparisons
5. **Theoretical limitations** (Condition 2) that severely restrict applicability

The paper should not be relied upon without independent verification of all results. Anyone building on this work should re-derive the analysis from scratch.
