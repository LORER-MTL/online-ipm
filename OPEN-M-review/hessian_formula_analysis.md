# Deep Analysis: The Hessian Formula Error in Lemma 2

## The Problem Statement

The paper works with a **reduced-space formulation** to handle equality constraints:

Given:
- Original problem: $\min_x f_t(x)$ subject to $A_t x = b_t$
- Constraint matrix $A_t \in \mathbb{R}^{p \times n}$ with $\text{rank}(A_t) = p < n$
- Null space basis $F_t \in \mathbb{R}^{n \times (n-p)}$ with orthonormal columns: $F_t^T F_t = I_{n-p}$
- Particular solution $\hat{x}_t$ satisfying $A_t \hat{x}_t = b_t$

The reduced function is:
$$\tilde{f}_t(z) = f_t(F_t z + \hat{x}_t)$$

where $z \in \mathbb{R}^{n-p}$ is the reduced variable.

---

## The Paper's Formula (WRONG)

**Their claim in Lemma 2:**
$$\nabla^2 \tilde{f}_t(z) = F_t \nabla^2 f_t(x_t^*) F_t^T$$

**Dimensions:**
- $F_t$: $n \times (n-p)$
- $\nabla^2 f_t(x_t^*)$: $n \times n$
- $F_t^T$: $(n-p) \times n$
- Result: $n \times n$ matrix ❌

**This is dimensionally wrong!** The Hessian of $\tilde{f}_t$ should be $(n-p) \times (n-p)$, not $n \times n$.

---

## The Correct Formula (with Derivation)

### Chain Rule for Vector Functions

For a composite function $g(z) = f(h(z))$ where:
- $z \in \mathbb{R}^m$
- $h: \mathbb{R}^m \to \mathbb{R}^n$
- $f: \mathbb{R}^n \to \mathbb{R}$

The gradient is:
$$\nabla g(z) = J_h(z)^T \nabla f(h(z))$$

where $J_h(z) \in \mathbb{R}^{n \times m}$ is the Jacobian of $h$.

The Hessian is:
$$\nabla^2 g(z) = J_h(z)^T \nabla^2 f(h(z)) J_h(z) + \sum_{i=1}^n \frac{\partial f}{\partial x_i}(h(z)) \nabla^2 h_i(z)$$

### Applying to Our Case

Here:
- $z \in \mathbb{R}^{n-p}$
- $h(z) = F_t z + \hat{x}_t \in \mathbb{R}^n$
- $f_t: \mathbb{R}^n \to \mathbb{R}$

The Jacobian of $h$:
$$J_h(z) = F_t \in \mathbb{R}^{n \times (n-p)}$$

Since $h(z)$ is **affine** (linear plus constant), $\nabla^2 h_i(z) = 0$ for all $i$. The second-order term vanishes.

Therefore:
$$\boxed{\nabla^2 \tilde{f}_t(z) = F_t^T \nabla^2 f_t(F_t z + \hat{x}_t) F_t}$$

**Dimensions:**
- $F_t^T$: $(n-p) \times n$
- $\nabla^2 f_t(\cdot)$: $n \times n$
- $F_t$: $n \times (n-p)$
- Result: $(n-p) \times (n-p)$ matrix ✓

---

## Step-by-Step Verification

### First Derivative (Gradient)

$$\nabla \tilde{f}_t(z) = F_t^T \nabla f_t(F_t z + \hat{x}_t)$$

**Proof:** By chain rule, for $j = 1, \ldots, n-p$:
$$\frac{\partial \tilde{f}_t}{\partial z_j} = \sum_{i=1}^n \frac{\partial f_t}{\partial x_i} \frac{\partial x_i}{\partial z_j} = \sum_{i=1}^n \frac{\partial f_t}{\partial x_i} F_{ij}$$

In matrix form: $\nabla \tilde{f}_t(z) = F_t^T \nabla f_t(x)$ where $x = F_t z + \hat{x}_t$. ✓

### Second Derivative (Hessian)

Taking the derivative of the gradient:
$$\frac{\partial}{\partial z_k} \left[ \nabla \tilde{f}_t(z) \right]_j = \frac{\partial}{\partial z_k} \left[ \sum_{i=1}^n F_{ij} \frac{\partial f_t}{\partial x_i}(x) \right]$$

Using chain rule again:
$$= \sum_{i=1}^n F_{ij} \sum_{\ell=1}^n \frac{\partial^2 f_t}{\partial x_i \partial x_\ell}(x) \frac{\partial x_\ell}{\partial z_k}$$

$$= \sum_{i=1}^n F_{ij} \sum_{\ell=1}^n \frac{\partial^2 f_t}{\partial x_i \partial x_\ell}(x) F_{\ell k}$$

In matrix form:
$$\left[ \nabla^2 \tilde{f}_t(z) \right]_{jk} = \sum_{i=1}^n \sum_{\ell=1}^n F_{ij}^T \left[\nabla^2 f_t(x)\right]_{i\ell} F_{\ell k}$$

$$= \left[ F_t^T \nabla^2 f_t(F_t z + \hat{x}_t) F_t \right]_{jk}$$

Therefore:
$$\boxed{\nabla^2 \tilde{f}_t(z) = F_t^T \nabla^2 f_t(F_t z + \hat{x}_t) F_t}$$

---

## Concrete Example 1: Simple Case

### Setup
- $n = 3$ (original space)
- $p = 1$ (one constraint)
- $n - p = 2$ (reduced space)

Constraint: $A x = [1, 1, 1] x = b$ (sum of components equals $b$)

Null space basis (orthonormal columns):
$$F = \begin{bmatrix} 
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\
-\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\
0 & -\frac{2}{\sqrt{6}}
\end{bmatrix}$$

Verify: $F^T F = I_2$ and $A F = 0$.

Particular solution: $\hat{x} = [b/3, b/3, b/3]^T$

### Example Function
$$f(x) = \frac{1}{2}(x_1^2 + x_2^2 + x_3^2)$$

Gradient:
$$\nabla f(x) = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}$$

Hessian:
$$\nabla^2 f(x) = I_3 = \begin{bmatrix} 
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}$$

### Reduced Function
$$\tilde{f}(z) = f(F z + \hat{x})$$

Let $z = [z_1, z_2]^T$. Then:
$$x = F z + \hat{x} = \begin{bmatrix} 
\frac{z_1}{\sqrt{2}} + \frac{z_2}{\sqrt{6}} + \frac{b}{3} \\
-\frac{z_1}{\sqrt{2}} + \frac{z_2}{\sqrt{6}} + \frac{b}{3} \\
-\frac{2z_2}{\sqrt{6}} + \frac{b}{3}
\end{bmatrix}$$

### Computing $\nabla^2 \tilde{f}(z)$ Directly

$$\tilde{f}(z) = \frac{1}{2} \left[ \left(\frac{z_1}{\sqrt{2}} + \frac{z_2}{\sqrt{6}} + \frac{b}{3}\right)^2 + \left(-\frac{z_1}{\sqrt{2}} + \frac{z_2}{\sqrt{6}} + \frac{b}{3}\right)^2 + \left(-\frac{2z_2}{\sqrt{6}} + \frac{b}{3}\right)^2 \right]$$

Expanding (the $b/3$ terms and cross-terms matter, but for the Hessian they vanish since it's second derivative):

$$\frac{\partial^2 \tilde{f}}{\partial z_1^2} = \frac{1}{2} \cdot 2 \cdot \frac{1}{2} = \frac{1}{2} + \frac{1}{2} = 1$$

$$\frac{\partial^2 \tilde{f}}{\partial z_2^2} = \frac{1}{2} \cdot 2 \cdot \left(\frac{1}{6} + \frac{1}{6} + \frac{4}{6}\right) = \frac{1}{2} \cdot 2 \cdot 1 = 1$$

$$\frac{\partial^2 \tilde{f}}{\partial z_1 \partial z_2} = 0$$

So:
$$\nabla^2 \tilde{f}(z) = I_2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

### Using the Correct Formula

$$\nabla^2 \tilde{f}(z) = F^T \nabla^2 f(Fz + \hat{x}) F = F^T I_3 F = F^T F = I_2$$

✓ **Matches!**

### Using the Paper's (Wrong) Formula

$$F \nabla^2 f(x^*) F^T = F I_3 F^T = F F^T$$

$$= \begin{bmatrix} 
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\
-\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\
0 & -\frac{2}{\sqrt{6}}
\end{bmatrix}
\begin{bmatrix} 
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0 \\
\frac{1}{\sqrt{6}} & \frac{1}{\sqrt{6}} & -\frac{2}{\sqrt{6}}
\end{bmatrix}$$

$$= \begin{bmatrix} 
\frac{1}{2} + \frac{1}{6} & -\frac{1}{2} + \frac{1}{6} & -\frac{2}{6} \\
-\frac{1}{2} + \frac{1}{6} & \frac{1}{2} + \frac{1}{6} & -\frac{2}{6} \\
-\frac{2}{6} & -\frac{2}{6} & \frac{4}{6}
\end{bmatrix}
= \begin{bmatrix} 
\frac{2}{3} & -\frac{1}{3} & -\frac{1}{3} \\
-\frac{1}{3} & \frac{2}{3} & -\frac{1}{3} \\
-\frac{1}{3} & -\frac{1}{3} & \frac{2}{3}
\end{bmatrix}$$

This is a $3 \times 3$ matrix, not $2 \times 2$! ❌

**Wrong dimension, wrong values!**

---

## Concrete Example 2: Non-Identity Hessian

### Setup
Same as Example 1, but with:
$$f(x) = x_1^2 + 2x_2^2 + 3x_3^2$$

Hessian:
$$\nabla^2 f(x) = \begin{bmatrix} 
2 & 0 & 0 \\
0 & 4 & 0 \\
0 & 0 & 6
\end{bmatrix} = \text{diag}(2, 4, 6)$$

### Correct Formula

$$\nabla^2 \tilde{f}(z) = F^T \text{diag}(2, 4, 6) F$$

$$= \begin{bmatrix} 
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0 \\
\frac{1}{\sqrt{6}} & \frac{1}{\sqrt{6}} & -\frac{2}{\sqrt{6}}
\end{bmatrix}
\begin{bmatrix} 
2 & 0 & 0 \\
0 & 4 & 0 \\
0 & 0 & 6
\end{bmatrix}
\begin{bmatrix} 
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\
-\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\
0 & -\frac{2}{\sqrt{6}}
\end{bmatrix}$$

First multiply $\text{diag}(2,4,6) F$:
$$= \begin{bmatrix} 
\frac{2}{\sqrt{2}} & \frac{2}{\sqrt{6}} \\
-\frac{4}{\sqrt{2}} & \frac{4}{\sqrt{6}} \\
0 & -\frac{12}{\sqrt{6}}
\end{bmatrix}
= \begin{bmatrix} 
\sqrt{2} & \frac{2}{\sqrt{6}} \\
-2\sqrt{2} & \frac{4}{\sqrt{6}} \\
0 & -2\sqrt{6}
\end{bmatrix}$$

Then multiply $F^T$ times this:
$$= \begin{bmatrix} 
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0 \\
\frac{1}{\sqrt{6}} & \frac{1}{\sqrt{6}} & -\frac{2}{\sqrt{6}}
\end{bmatrix}
\begin{bmatrix} 
\sqrt{2} & \frac{2}{\sqrt{6}} \\
-2\sqrt{2} & \frac{4}{\sqrt{6}} \\
0 & -2\sqrt{6}
\end{bmatrix}$$

$$= \begin{bmatrix} 
1 + 2 & \frac{2}{6} - \frac{4}{6} \\
\frac{2}{6} - \frac{4}{6} & \frac{2}{6} + \frac{4}{6} + \frac{24}{6}
\end{bmatrix}
= \begin{bmatrix} 
3 & -\frac{1}{3} \\
-\frac{1}{3} & 5
\end{bmatrix}$$

This is a $2 \times 2$ matrix. ✓

### Paper's (Wrong) Formula

$$F \text{diag}(2, 4, 6) F^T$$

This gives a $3 \times 3$ matrix (dimensions: $3 \times 3 \times 3 \times 3 = 3 \times 3$).

Computing:
$$= \begin{bmatrix} 
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\
-\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\
0 & -\frac{2}{\sqrt{6}}
\end{bmatrix}
\begin{bmatrix} 
2 & 0 & 0 \\
0 & 4 & 0 \\
0 & 0 & 6
\end{bmatrix}
\begin{bmatrix} 
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0 \\
\frac{1}{\sqrt{6}} & \frac{1}{\sqrt{6}} & -\frac{2}{\sqrt{6}}
\end{bmatrix}$$

I'll spare the full calculation, but this is clearly the wrong dimension! ❌

---

## Why the Paper's Final Bounds Are Still Correct

The paper uses bounds like:
$$\|\nabla^2 \tilde{f}_t(z)\| \leq L'$$

Even though their formula is wrong, the bound can still be valid due to:

### Key Fact: Orthonormal $F_t$

Since $F_t^T F_t = I$, we have $\|F_t\| = 1$ (operator norm).

**Correct bound:**
$$\|\nabla^2 \tilde{f}_t(z)\| = \|F_t^T \nabla^2 f_t(x) F_t\| \leq \|F_t^T\| \|\nabla^2 f_t(x)\| \|F_t\| = 1 \cdot \|\nabla^2 f_t(x)\| \cdot 1$$

**Their (wrong) formula bound:**
$$\|F_t \nabla^2 f_t(x) F_t^T\| \leq \|F_t\| \|\nabla^2 f_t(x)\| \|F_t^T\| = 1 \cdot \|\nabla^2 f_t(x)\| \cdot 1$$

The bounds are the same! But:
1. Their formula gives the wrong dimension
2. Their formula evaluates to a different matrix
3. The bound happens to be the same due to $\|F_t\| = 1$

### Another Way to See It

The eigenvalues of $F^T H F$ are a subset of the eigenvalues of $H$ (for symmetric $H$).

Similarly, $F H F^T$ is a different matrix but its nonzero eigenvalues are related to those of $H$.

For $\|F\| = 1$, both satisfy:
$$\|F^T H F\| \leq \|H\|, \quad \|F H F^T\| \leq \|H\|$$

So the **spectral norm bounds** are the same, even though the matrices are different.

---

## Does the Correct Formula Fix Other Errors?

Let me check if using the correct Hessian formula would fix the other issues...

### Issue 1.3: Incorrect Inverse Formula

**Their claim:**
$$\nabla^2 \tilde{f}_t(z^*)^{-1} = F_t(F_t^T F_t)^{-1} \nabla^2 f_t(x_t^*)^{-1} (F_t^T F_t)^{-1} F_t^T$$

This is attempting to invert their wrong formula $F_t \nabla^2 f_t(x_t^*) F_t^T$.

**Correct approach:**

If $\nabla^2 \tilde{f}_t(z) = F_t^T \nabla^2 f_t(x) F_t$, then inverting (assuming invertibility):

Using the matrix identity: If $A = B^T C B$ where $B$ has full column rank and $C$ is invertible, then:
$$A^{-1} = (B^T C B)^{-1} = B^{-1} C^{-1} (B^T)^{-1}$$

But wait, $F_t$ is not square! We can't invert it directly.

**Correct inverse (using pseudo-inverse properties):**

For $F_t \in \mathbb{R}^{n \times (n-p)}$ with $F_t^T F_t = I_{n-p}$:

$$\left[ F_t^T H F_t \right]^{-1} = F_t^T H^{-1} F_t$$

**only if** $H$ is invertible on the range of $F_t$.

Actually, this is not trivial. Let me think more carefully.

If $\nabla^2 \tilde{f}_t(z) = F_t^T H F_t$ where $H = \nabla^2 f_t(x)$, and we want the inverse:

Let $A = F_t^T H F_t$. We want $A^{-1}$.

Let $w = F_t^T H F_t v$ for some $v \in \mathbb{R}^{n-p}$.

Then $F_t w = F_t F_t^T H F_t v = H F_t v$ (using $F_t F_t^T$ is projection onto range of $F_t$).

Wait, this is getting complicated. The key point is:

**The paper's inverse formula is wrong because:**
1. Their Hessian formula is wrong
2. Even trying to fix it, the algebra they wrote doesn't work for non-square matrices
3. $(F^T H F)^{-1} \neq F (F^T F)^{-1} H^{-1} (F^T F)^{-1} F^T$ in general

### So the Correct Formula Doesn't Fix This Error

Using the correct Hessian formula highlights that their inverse formula is even more wrong than I initially thought. The error compounds.

### Issue 1.6: Condition 2 Forces $v \approx 0$

This error is in the theorem statement and doesn't depend on the Hessian formula—it's a logical error in how they set up the conditions. Using the correct Hessian formula wouldn't fix it.

### Issue 1.4: Equation (9) Missing Square

This is about Newton's method convergence, which uses the Hessian. Let me check...

The Newton convergence bound is:
$$\|x_{k+1} - x^*\| \leq \frac{L}{2h} \|x_k - x^*\|^2$$

This depends on:
- $h$: lower bound on smallest eigenvalue of Hessian (strong convexity)
- $L$: Lipschitz constant of Hessian

Using the correct vs wrong Hessian formula:
- The **eigenvalue bounds** are the same (as shown above)
- So the constants $h, L$ for the reduced problem are related to the original problem in the same way
- **The missing square is still missing**—this is a separate error

### Issue 1.5: KKT Invertibility Condition

This is about the KKT system:
$$\begin{bmatrix} \nabla^2 f & A^T \\ A & 0 \end{bmatrix}$$

This involves the **original** Hessian $\nabla^2 f$, not the reduced Hessian $\nabla^2 \tilde{f}$.

The error is that they claim "invertible Hessian implies invertible KKT matrix," which is false.

**Using the correct reduced Hessian formula doesn't affect this—it's a separate error.**

---

## Summary of Findings

### The Correct Formula

$$\boxed{\nabla^2 \tilde{f}_t(z) = F_t^T \nabla^2 f_t(F_t z + \hat{x}_t) F_t}$$

where:
- $F_t \in \mathbb{R}^{n \times (n-p)}$ has orthonormal columns
- $\nabla^2 f_t(\cdot) \in \mathbb{R}^{n \times n}$ is the Hessian in the original space
- Result: $\nabla^2 \tilde{f}_t(z) \in \mathbb{R}^{(n-p) \times (n-p)}$ ✓

### What's Wrong with the Paper's Formula

$$\nabla^2 \tilde{f}_t(z) = F_t \nabla^2 f_t(x_t^*) F_t^T \quad \text{(WRONG)}$$

Problems:
1. **Wrong dimension:** $n \times n$ instead of $(n-p) \times (n-p)$
2. **Wrong matrix:** Even the nonzero part doesn't match the correct formula
3. **Missing dependence on $z$:** Evaluated at wrong point (should be at $F_t z + \hat{x}_t$, not $x_t^*$)

### Why Their Bounds Still Work

The spectral norm satisfies:
$$\|F_t^T H F_t\| \leq \|H\|, \quad \|F_t H F_t^T\| \leq \|H\|$$

when $\|F_t\| = 1$ (orthonormal columns).

So their **final numerical bounds** (inequalities) are still valid, even though the **formula** is wrong.

### Does Fixing This Fix Other Errors?

**No.** The other errors are independent:

- **Issue 1.3 (Inverse formula):** Still wrong; the correct Hessian formula makes this error more obvious
- **Issue 1.4 (Missing square):** Separate error in stating Newton convergence
- **Issue 1.5 (KKT condition):** About the original problem, not the reduced problem
- **Issue 1.6 (Condition 2):** Logical error in theorem statement

### The Deep Issue

The paper's authors appear to have:
1. **Confused the order of multiplication** ($F H F^T$ vs $F^T H F$)
2. **Not checked dimensions** (should have caught that $n \times n \neq (n-p) \times (n-p)$)
3. **Used correct bounds anyway** (possibly copied from another source, or got lucky)
4. **Not verified their formula with even a simple example**

This suggests:
- Lack of careful derivation
- Possible confusion about transpose operations
- The rest of the paper may have been built on this wrong formula, with compensating errors

---

## Recommendation

Anyone implementing this algorithm should:
1. **Use the correct formula:** $\nabla^2 \tilde{f}_t(z) = F_t^T \nabla^2 f_t(F_t z + \hat{x}_t) F_t$
2. **Verify dimensions** at every step
3. **Test with simple examples** (like the ones above)
4. **Be skeptical** of the paper's other derivations
5. **Independently verify** any bounds or convergence claims

The fact that such a fundamental error appears in a **lemma** (which should be carefully proved) raises serious questions about the rigor of the entire paper.
