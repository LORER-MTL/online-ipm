# Non-Differentiability Analysis: OPEN-M Numerical Example

## The Problem

The authors use the cost function:
$$f_i(x) = \alpha_i e^{\beta_i |x|}$$

where $|x|$ is the **absolute value** function, which is **not differentiable at $x = 0$**.

---

## 1. What Actually Breaks

### 1.1 Gradient Doesn't Exist at $x = 0$

For $x \neq 0$:
$$\frac{d}{dx} f_i(x) = \alpha_i e^{\beta_i |x|} \cdot \beta_i \cdot \text{sgn}(x)$$

where $\text{sgn}(x) = \begin{cases} +1 & x > 0 \\ -1 & x < 0 \\ \text{undefined} & x = 0 \end{cases}$

**At $x = 0$:**
- Left derivative: $\lim_{h \to 0^-} \frac{f_i(h) - f_i(0)}{h} = -\alpha_i \beta_i$
- Right derivative: $\lim_{h \to 0^+} \frac{f_i(h) - f_i(0)}{h} = +\alpha_i \beta_i$
- These don't match (unless $\beta_i = 0$), so **the gradient is undefined**

### 1.2 Hessian Doesn't Exist

For $x \neq 0$:
$$\frac{d^2}{dx^2} f_i(x) = \alpha_i e^{\beta_i |x|} \cdot \beta_i^2$$

This looks continuous, but it's **not twice differentiable** because the first derivative has a jump discontinuity at $x = 0$.

**Hessian at $x = 0$:**
- Doesn't exist in the classical sense
- Even as a generalized derivative (distributional sense), it involves a Dirac delta: $\beta_i^2 \alpha_i + 2\beta_i \alpha_i \delta(x)$

### 1.3 Newton's Method Cannot Run

Newton's method requires:
$$x_{k+1} = x_k - (\nabla^2 f_t(x_k))^{-1} \nabla f_t(x_k)$$

**If any component $x_k^{(i)} = 0$:**
- The gradient vector has undefined entries
- The Hessian matrix has undefined entries
- **The algorithm literally cannot execute**

---

## 2. Which Assumptions Break

### Assumption 2 (Lipschitz Continuous Hessian)

**Stated:** "The Hessian $\nabla^2 f_t(x)$ is Lipschitz continuous with constant $L$"

$$\|\nabla^2 f_t(x) - \nabla^2 f_t(y)\| \leq L\|x - y\|$$

**Why it breaks:**
1. The Hessian **doesn't exist** at $x = 0$, so you can't even define $\nabla^2 f_t(0)$
2. Even if you extend it by continuity, it's not Lipschitz:
   - Consider $x = \epsilon$ and $y = -\epsilon$ near 0
   - $\nabla f(\epsilon) = \alpha \beta e^{\beta \epsilon}$ (positive)
   - $\nabla f(-\epsilon) = -\alpha \beta e^{\beta \epsilon}$ (negative)
   - Jump: $|\nabla f(\epsilon) - \nabla f(-\epsilon)| = 2\alpha \beta e^{\beta \epsilon}$
   - Distance: $|\epsilon - (-\epsilon)| = 2\epsilon$
   - Ratio: $\frac{2\alpha \beta e^{\beta \epsilon}}{2\epsilon} \to \infty$ as $\epsilon \to 0$

The gradient is **not even Lipschitz**, let alone the Hessian.

### Claim of Twice-Differentiability (Section IV)

**Stated:** "convex, twice-differentiable functions"

This is **false**. The function $f_i(x) = \alpha_i e^{\beta_i |x|}$ is:
- ✓ Convex (for $\alpha_i, \beta_i > 0$)
- ✗ **Not differentiable** at $x = 0$ (gradient undefined)
- ✗ **Not twice-differentiable** anywhere (first derivative discontinuous)

### Lemma 3 (Newton Convergence)

**Requires:** Smooth function with Lipschitz Hessian

Newton's method quadratic convergence proof relies on:
$$\|\nabla f(x) - \nabla f(x^*) - \nabla^2 f(x^*)(x - x^*)\| \leq \frac{L}{2}\|x - x^*\|^2$$

This requires $\nabla^2 f$ to exist and be continuous. **Fails completely** for the chosen cost function.

---

## 3. Can They Actually Get the Claimed Results?

### 3.1 Theoretical Answer: No

Under the stated problem formulation with $f_i(x) = \alpha_i e^{\beta_i |x|}$:
- The algorithm **cannot run** if any iterate has a component equal to 0
- The convergence analysis (Lemmas 2, 3, Theorems 1, 2) **does not apply**
- The dynamic regret bounds **are not valid**

### 3.2 What Likely Happened in Practice

Several possibilities:

#### Option A: They Got Lucky
- The optimal solutions $x_t^*$ never hit exactly 0
- The iterates $x_k$ never hit exactly 0
- This is **measure-theoretically unlikely** for a network flow problem where some arcs might have zero flow

#### Option B: Numerical Regularization
- Floating-point arithmetic might avoid exact zeros
- Computer implementations use finite precision: $x \approx 10^{-16}$ rather than $x = 0$
- But this is **not mentioned** and **doesn't justify the theory**

#### Option C: They Actually Used a Different Function
- They might have implemented a smoothed version: $f_i(x) = \alpha_i e^{\beta_i \sqrt{x^2 + \delta^2}}$ for small $\delta > 0$
- Or used the smooth approximation: $|x| \approx \sqrt{x^2 + \delta^2}$ or $|x| \approx x \tanh(x/\delta)$
- **Not mentioned in the paper**

#### Option D: They Used Subgradient Methods
- At non-differentiable points, use any subgradient from $\partial f_i(0) = \alpha_i \beta_i [-1, 1]$
- But this is **not Newton's method** anymore
- Convergence theory changes completely

### 3.3 The Network Flow Context Makes It Worse

In optimal power flow / network flow:
- Flow variables can easily be zero (some arcs unused)
- At optimality, many arcs might have zero flow (especially in tree networks)
- The probability of hitting $x_i = 0$ is **high**, not negligible

For a radial network (tree), many flows will be zero at boundaries.

---

## 4. Non-Differentiability in Newton's Method Itself

### 4.1 Classical Newton's Method Requires Smoothness

**Standard requirements for Newton convergence:**
1. $f$ is twice continuously differentiable
2. $\nabla^2 f(x^*)$ is non-singular (positive definite for minimization)
3. Initial point $x_0$ is sufficiently close to $x^*$

**What happens without smoothness:**
- No guarantee of descent
- No quadratic convergence
- Algorithm may not even be well-defined

### 4.2 Constrained Newton (Equality Constraints)

For the problem:
$$\min_{x} f(x) \quad \text{s.t.} \quad Ax = b$$

Newton's method solves the KKT system:
$$\begin{bmatrix} \nabla^2 f(x_k) & A^\top \\ A & 0 \end{bmatrix} \begin{bmatrix} \Delta x \\ \lambda \end{bmatrix} = \begin{bmatrix} -\nabla f(x_k) \\ 0 \end{bmatrix}$$

**Required:**
- $\nabla^2 f(x_k)$ must exist (needs twice-differentiability)
- $\nabla f(x_k)$ must exist (needs differentiability)

At $x = 0$ for their cost function: **both fail**.

### 4.3 The Projection Step

OPEN-M first projects onto the new constraint:
$$x_t' = \arg\min_x \|x - x_{t-1}\|^2 \quad \text{s.t.} \quad A_t x = b_t$$

This is a **quadratic program** (smooth), so the projection step is fine.

**But then Newton's method:**
$$x_t = x_t' - (\nabla^2 f_t(x_t'))^{-1} \nabla f_t(x_t')$$

If $x_t'$ has a component equal to 0, this step **cannot be computed**.

### 4.4 Reduced Space Formulation

The paper also discusses working in the null space:
$$\tilde{f}_t(z) = f_t(F_t z + \hat{x})$$

where $F_t$ has columns spanning $\mathcal{N}(A_t)$.

**Chain rule for gradient:**
$$\nabla \tilde{f}_t(z) = F_t^\top \nabla f_t(F_t z + \hat{x})$$

**If $F_t z + \hat{x}$ has a zero component:**
- $\nabla f_t(F_t z + \hat{x})$ is undefined
- $\nabla \tilde{f}_t(z)$ is undefined
- Newton's method in reduced space **fails**

---

## 5. Correct Cost Functions for the Experiment

### 5.1 What They Should Have Used

For a smooth, convex cost function in network flow:

**Option 1: Quadratic**
$$f_i(x) = \alpha_i x^2 + \beta_i x + \gamma_i$$

**Option 2: Smooth Exponential**
$$f_i(x) = \alpha_i e^{\beta_i x^2}$$

**Option 3: Log-Barrier (for $x > 0$)**
$$f_i(x) = -\alpha_i \log(x) + \beta_i x$$

**Option 4: Smoothed Absolute Value**
$$f_i(x) = \alpha_i \sqrt{x^2 + \delta^2}$$
for small $\delta > 0$ (Huber-like)

All are smooth, twice-differentiable, and support Newton's method.

### 5.2 Why They Might Have Chosen $|x|$

Possible reasons:
1. **Physical interpretation**: Absolute value models losses in power systems ($I^2 R$ losses)
2. **Common in OPF literature**: Linearizations use $|x|$ for piecewise linear costs
3. **Copy-paste error**: Took a formula from a paper using subgradient methods

But **none of this justifies using it with Newton's method**.

---

## 6. Impact on Paper's Validity

### 6.1 Numerical Results (Section IV)

**Verdict: Unreliable**

- The results are either:
  - (A) Obtained by numerical luck (never hitting 0 exactly)
  - (B) Using a different, undisclosed implementation
  - (C) Using subgradients, not Newton's method

**None of these match the described algorithm.**

### 6.2 Theoretical Results (Lemmas, Theorems)

**Verdict: Not affected directly**

The theoretical analysis assumes:
- Smoothness (Assumption 2)
- Twice-differentiability

The analysis is **correct under these assumptions**, but:
- The numerical example **violates the assumptions**
- The theory **does not apply** to the example
- This is a **major inconsistency**

### 6.3 Algorithm Description

**Verdict: Incomplete**

The paper should either:
1. Use a smooth cost function in experiments
2. Or explain how to handle non-smoothness (e.g., smoothing, subgradients)
3. Or prove convergence for non-smooth objectives (much harder)

**Current state:** Theory and experiments are **incompatible**.

---

## 7. How to Fix the Paper

### Fix 1: Change the Cost Function (Easy)

Replace $f_i(x) = \alpha_i e^{\beta_i |x|}$ with:
$$f_i(x) = \alpha_i e^{\beta_i x^2}$$

- Still convex
- Now smooth
- Theory applies

### Fix 2: Smooth the Absolute Value (Medium)

Use:
$$f_i(x) = \alpha_i e^{\beta_i \sqrt{x^2 + \delta^2}}$$

for $\delta = 10^{-6}$ or similar.

- Approximates $|x|$ closely
- Smooth everywhere
- Document the approximation

### Fix 3: Extend Theory to Non-Smooth (Hard)

Develop a version using:
- Proximal Newton methods
- Generalized derivatives
- Bundle methods

**This would be novel research**, not a simple fix.

---

## 8. Comparison to Related Work

### 8.1 Interior Point Methods (Reference [9])

Interior point methods handle inequality constraints:
$$\min_x f(x) \quad \text{s.t.} \quad Ax = b, \; x \geq 0$$

Using barrier: $\phi(x) = f(x) - \mu \sum_i \log(x_i)$

**Key difference:**
- The barrier $-\log(x_i)$ is smooth on $x > 0$
- It automatically **prevents $x_i = 0$** (barrier goes to $+\infty$)
- Newton's method is **well-defined** throughout

The authors' use of a **non-smooth objective without barriers** is the problem.

### 8.2 Subgradient Methods

Online subgradient descent handles non-smooth objectives:
$$x_{t+1} = \Pi_{\mathcal{C}}(x_t - \eta_t g_t)$$

where $g_t \in \partial f_t(x_t)$ is any subgradient.

**For $f_i(x) = \alpha_i e^{\beta_i |x|}$:**
$$\partial f_i(0) = [\alpha_i \beta_i e^{0} \cdot (-1, +1)] = \alpha_i \beta_i [-1, +1]$$

This works, but:
- Only **linear** convergence (not quadratic)
- Needs step size tuning
- **Not Newton's method**

---

## 9. Conclusion

### The Non-Differentiability Issue is Critical Because:

1. **Algorithm cannot execute**: If any iterate has a zero component, Newton's method is undefined

2. **Theory completely invalid**: Assumption 2 (Lipschitz Hessian) fails, so all convergence results are void for this example

3. **Results are unexplained**: Either:
   - Pure luck (unlikely)
   - Different implementation than described (dishonest)
   - Numerical artifacts (unstated)

4. **Shows lack of rigor**: The authors didn't verify their cost function satisfies their own assumptions

### This is More Than a Technicality

Unlike some mathematical assumptions (e.g., assuming a parameter is bounded when it's actually unbounded), this is a **showstopper**:

- Newton's method **literally cannot run** on non-differentiable functions
- This is not an edge case—it's the **stated objective** of the experiment
- The mismatch between theory and experiment is complete

### Recommendation

The numerical example (Section IV) should be **entirely redone** with a smooth cost function, or the paper should be clear that:
1. They used a smoothed approximation (and provide details)
2. The theory doesn't apply to the experiment as written
3. Empirical results are exploratory, not validation of the theory

As written, Section IV **does not validate the theoretical results** and should not be trusted.
