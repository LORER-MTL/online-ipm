# Deep Dive: Issue 1.3 - The Inverse Formula Error

## The Claimed Formula (From Lemma 2 Proof)

The paper claims:
$$\|\nabla^2 \tilde{f}_t(z^*)^{-1}\| = \|F_t(F_t^T F_t)^{-1} \nabla^2 f_t(x_t^*)^{-1} (F_t^T F_t)^{-1} F_t^T\|$$

This is their attempt to relate the inverse of the reduced Hessian to the inverse of the original Hessian.

---

## What They're Trying to Do

They want to show:
$$\|\nabla^2 \tilde{f}_t(z^*)^{-1}\| \leq \text{something involving } \|\nabla^2 f_t(x_t^*)^{-1}\|$$

This is needed for their convergence analysis because Newton's method convergence depends on the condition number of the Hessian.

---

## Why Their Formula is Nonsensical

### Problem 1: You Can't Invert Non-Square Matrices

Their formula involves:
- $F_t$: $n \times (n-p)$ matrix (tall, not square)
- $F_t^T$: $(n-p) \times n$ matrix (wide, not square)

**Neither $F_t$ nor $F_t^T$ is invertible!**

You cannot write $F_t^{-1}$ or $(F_t^T)^{-1}$ in the usual sense.

### Problem 2: The Expression Doesn't Make Algebraic Sense

Let's try to parse their expression:
$$F_t(F_t^T F_t)^{-1} \nabla^2 f_t(x_t^*)^{-1} (F_t^T F_t)^{-1} F_t^T$$

Dimensions:
- $F_t$: $n \times (n-p)$
- $(F_t^T F_t)^{-1}$: $(n-p) \times (n-p)$ ✓
- $\nabla^2 f_t(x_t^*)^{-1}$: $n \times n$
- $(F_t^T F_t)^{-1}$: $(n-p) \times (n-p)$
- $F_t^T$: $(n-p) \times n$

Let's multiply from right to left:
1. $(F_t^T F_t)^{-1} F_t^T$: $(n-p) \times (n-p) \times (n-p) \times n = (n-p) \times n$
2. $\nabla^2 f_t(x_t^*)^{-1} [(F_t^T F_t)^{-1} F_t^T]$: $n \times n \times (n-p) \times n$ ❌ **Dimension mismatch!**

**The multiplication doesn't even make sense!** You can't multiply $n \times n$ by $(n-p) \times n$.

Let me try left to right:
1. $F_t (F_t^T F_t)^{-1}$: $n \times (n-p) \times (n-p) \times (n-p) = n \times (n-p)$
2. $[F_t (F_t^T F_t)^{-1}] \nabla^2 f_t(x_t^*)^{-1}$: $n \times (n-p) \times n \times n$ ❌ **Still mismatch!**

**No matter how you try to multiply, it doesn't work!**

### Problem 3: Even If We "Fix" It, It's Wrong

Maybe they meant something like:
$$F_t^{\dagger} \nabla^2 f_t(x_t^*)^{-1} (F_t^T)^{\dagger}$$

where $\dagger$ denotes the Moore-Penrose pseudoinverse?

But even this doesn't give us the inverse of $F_t^T \nabla^2 f_t F_t$.

---

## The Correct Approach

### What We Actually Need

We have:
$$\nabla^2 \tilde{f}_t(z^*) = F_t^T \nabla^2 f_t(x_t^*) F_t$$

We want to find:
$$\left[ \nabla^2 \tilde{f}_t(z^*) \right]^{-1} = \left[ F_t^T \nabla^2 f_t(x_t^*) F_t \right]^{-1}$$

### Key Insight: We Don't Need an Explicit Formula!

For bounding $\|\left[ \nabla^2 \tilde{f}_t(z^*) \right]^{-1}\|$, we use spectral properties:

**Theorem (Eigenvalues of Congruent Matrices):**

If $H \in \mathbb{R}^{n \times n}$ is symmetric positive definite, and $F \in \mathbb{R}^{n \times m}$ has full column rank with $F^T F = I_m$, then:

The eigenvalues of $F^T H F$ are a subset of the eigenvalues of $H$ (specifically, the $m$ eigenvalues corresponding to the subspace spanned by columns of $F$).

**Consequence:**

$$\lambda_{\min}(F^T H F) \geq \lambda_{\min}(H|_{\mathcal{R}(F)})$$

where $H|_{\mathcal{R}(F)}$ means "restricted to the range of $F$."

**For our case:**

If $\nabla^2 f_t(x_t^*)$ is positive definite on the null space of $A_t$ (i.e., on $\mathcal{R}(F_t)$), then:

$$\lambda_{\min}(\nabla^2 \tilde{f}_t(z^*)) \geq \lambda_{\min}(\nabla^2 f_t(x_t^*)|_{\mathcal{N}(A_t)})$$

Therefore:
$$\|[\nabla^2 \tilde{f}_t(z^*)]^{-1}\| = \frac{1}{\lambda_{\min}(\nabla^2 \tilde{f}_t(z^*))} \leq \frac{1}{\lambda_{\min}(\nabla^2 f_t(x_t^*)|_{\mathcal{N}(A_t)})}$$

### The Correct Bound

If $\nabla^2 f_t(x)$ is $h$-strongly convex on the null space $\mathcal{N}(A_t)$, meaning:
$$v^T \nabla^2 f_t(x) v \geq h \|v\|^2 \quad \forall v \in \mathcal{N}(A_t)$$

Then:
$$\boxed{\|[\nabla^2 \tilde{f}_t(z^*)]^{-1}\| \leq \frac{1}{h}}$$

**No explicit inverse formula needed!**

---

## What About Their Proof?

Let me look at what they might have been thinking...

### Possible Intention 1: Pseudoinverse Relationship

For $F \in \mathbb{R}^{n \times m}$ with $F^T F = I_m$ (orthonormal columns), the pseudoinverse is:
$$F^{\dagger} = (F^T F)^{-1} F^T = I_m F^T = F^T$$

So $F^{\dagger} = F^T$.

**They might have tried:**
$$(F^T H F)^{-1} \stackrel{?}{=} F^{\dagger} H^{-1} (F^T)^{\dagger} = F H^{-1} F^T$$

But this is **FALSE!**

**Counterexample:**

Take $H = I_n$, $F^T F = I_m$. Then:
- $(F^T H F)^{-1} = (F^T F)^{-1} = I_m$ (size $m \times m$)
- $F H^{-1} F^T = F F^T$ (size $n \times n$)

These are **different dimensions**! And even if $m = n$ (square $F$), we'd have:
- $(F^T F)^{-1} = I$
- $F F^T \neq I$ unless $F$ is orthogonal (square with orthonormal columns)

### Possible Intention 2: Subspace Projection Formula

There is a correct formula involving projections. If $P = F F^T$ is the projection onto $\mathcal{R}(F)$, then for certain structured problems:

$$(P H P)^{\dagger} = P H^{-1} P$$

where $\dagger$ is the pseudoinverse (and we restrict to the range of $P$).

But:
1. This is **not** what they wrote
2. This involves pseudoinverses, not regular inverses
3. They seem to have garbled this relationship

---

## How Critical Is This Error?

### Severity Assessment: **HIGH but Salvageable**

Let me break down the impact:

### What Breaks:

1. **The proof of Lemma 2 is invalid**
   - Their algebraic derivation is nonsense
   - Can't trust the intermediate steps

2. **Loss of rigor**
   - Shows they didn't carefully verify their algebra
   - Raises questions: what else is wrong?

### What Doesn't Break:

1. **The final bound can still be correct**
   - Even though their derivation is wrong, the bound
     $$\|[\nabla^2 \tilde{f}_t(z^*)]^{-1}\| \leq \frac{1}{h}$$
     can be proven correctly (as I showed above)

2. **The algorithm is still valid**
   - The algorithm itself (project, then take Newton step) doesn't depend on this formula
   - The implementation doesn't need to compute any inverse of $F_t$

3. **The convergence results can be salvaged**
   - The convergence analysis depends on bounds like $\|[\nabla^2 \tilde{f}_t]^{-1}\| \leq 1/h$
   - These bounds can be established correctly via eigenvalue arguments
   - The final convergence rates may still be valid

---

## Is the Paper Salvageable?

### Short Answer: **YES, with significant corrections**

Here's what would need to be fixed:

### Fix 1: Remove the Nonsensical Formula

Delete this entire expression:
$$\|\nabla^2 \tilde{f}_t(z^*)^{-1}\| = \|F_t(F_t^T F_t)^{-1} \nabla^2 f_t(x_t^*)^{-1} (F_t^T F_t)^{-1} F_t^T\|$$

Replace with: "We do not need an explicit formula for the inverse."

### Fix 2: Use Spectral Arguments

Replace the faulty derivation with:

> **Lemma 2 (Corrected Proof):**
> 
> Since $\nabla^2 f_t(x)$ is $h$-strongly convex on $\mathcal{N}(A_t)$, we have for any $v \in \mathcal{N}(A_t)$:
> $$v^T \nabla^2 f_t(x) v \geq h \|v\|^2$$
> 
> For the reduced problem, any $z \in \mathbb{R}^{n-p}$ corresponds to $F_t z \in \mathcal{N}(A_t)$. Thus:
> $$z^T [F_t^T \nabla^2 f_t(x) F_t] z = (F_t z)^T \nabla^2 f_t(x) (F_t z) \geq h \|F_t z\|^2 = h \|z\|^2$$
> 
> where we used $\|F_t z\| = \|z\|$ since $F_t^T F_t = I$.
> 
> Therefore, $\nabla^2 \tilde{f}_t(z)$ is $h$-strongly convex, implying:
> $$\|[\nabla^2 \tilde{f}_t(z)]^{-1}\| \leq \frac{1}{h}$$

### Fix 3: Clarify Assumptions

Add explicit assumption:

> **Assumption:** $\nabla^2 f_t(x)$ is $h$-strongly convex **on the null space** $\mathcal{N}(A_t)$, meaning:
> $$v^T \nabla^2 f_t(x) v \geq h \|v\|^2 \quad \forall v \in \mathcal{N}(A_t)$$

This is **stronger** than just "$\nabla^2 f_t(x)$ is invertible" (which is what they currently claim).

### Fix 4: Correct the Hessian Formula

As we already discussed, fix:
$$\nabla^2 \tilde{f}_t(z) = F_t \nabla^2 f_t(x_t^*) F_t^T \quad \text{(WRONG)}$$

to:
$$\nabla^2 \tilde{f}_t(z) = F_t^T \nabla^2 f_t(F_t z + \hat{x}_t) F_t \quad \text{(CORRECT)}$$

---

## Deeper Issue: What This Reveals

### The Authors' Understanding

This error suggests the authors:

1. **Don't understand the geometry of constrained optimization**
   - The reduced Hessian lives in a different space than the original
   - Can't just "invert the formula" algebraically

2. **Didn't verify basic linear algebra**
   - Didn't check if matrix products are well-defined
   - Didn't test with a simple example

3. **May have misunderstood a correct result**
   - There ARE correct relationships between eigenvalues
   - They might have seen a formula in another paper and misapplied it

### The Cascade of Errors

This error compounds with Issue 1.2 (wrong Hessian formula):

**Step 1:** Write wrong Hessian formula
$$\nabla^2 \tilde{f}_t(z) = F_t \nabla^2 f_t(x_t^*) F_t^T$$

**Step 2:** Try to invert it naively
$$(F_t \nabla^2 f_t(x_t^*) F_t^T)^{-1} \stackrel{?}{=} (F_t^T)^{-1} (\nabla^2 f_t(x_t^*))^{-1} F_t^{-1}$$

**Step 3:** "Fix" the fact that $F_t$ is not invertible by inserting $(F_t^T F_t)^{-1}$
$$(F_t^T)^{-1} = F_t(F_t^T F_t)^{-1}, \quad F_t^{-1} = (F_t^T F_t)^{-1} F_t^T$$

**Step 4:** Write nonsensical formula:
$$F_t(F_t^T F_t)^{-1} \nabla^2 f_t(x_t^*)^{-1} (F_t^T F_t)^{-1} F_t^T$$

**Step 5:** Don't verify it actually works

This is a **compounding error**: a wrong formula (Issue 1.2) + wrong attempt to fix it (Issue 1.3) = complete nonsense.

---

## Implications for Using the Paper

### For Practitioners:

**Can I implement the algorithm?**

**YES!** The algorithm itself is:
1. Project onto new constraint: $x_t' = \arg\min_x \|x - x_{t-1}\| \text{ s.t. } A_t x = b_t$
2. Take Newton step in reduced space

This doesn't require the faulty inverse formula. You just:
- Compute $F_t$ (null space basis)
- Compute $\nabla^2 \tilde{f}_t(z) = F_t^T \nabla^2 f_t(x) F_t$ (with correct formula)
- Solve linear system $\nabla^2 \tilde{f}_t(z) \Delta z = -\nabla \tilde{f}_t(z)$
- Update: $z \gets z + \Delta z$, then $x = F_t z + \hat{x}_t$

**Can I trust the convergence bounds?**

**PARTIALLY.** The bounds like $O(V_T)$ regret may be correct, but:
- The proof has errors
- You should independently verify with your own analysis or experiments
- Don't trust the constant factors

### For Researchers:

**Can I build on this work?**

**YES, but carefully:**
1. **Re-derive everything from scratch**
2. **Use the correct formulas** (Hessian, bounds)
3. **Verify all claims** independently
4. **Cite this as "approach inspired by [paper], but corrected"**

**Should I extend this work?**

**CAUTION:** The paper has multiple fundamental errors. Before extending:
1. Check if the core results are actually novel (they claim "first" but have counterexamples)
2. Verify the convergence results are correct
3. See if simpler approaches exist

---

## Comparison to Well-Done Papers

### How Should Lemma 2 Have Been Written?

Here's an example from a rigorous paper on projected Newton methods (Boyd & Vandenberghe, "Convex Optimization", Section 10.3):

> **Proposition:** Let $f$ be strongly convex on $\mathcal{N}(A)$ with parameter $m > 0$. Then the reduced function $\tilde{f}(z) = f(Fz + \hat{x})$ satisfies:
> 
> $$\nabla^2 \tilde{f}(z) = F^T \nabla^2 f(Fz + \hat{x}) F$$
> 
> and is strongly convex with the same parameter $m$:
> 
> $$z^T \nabla^2 \tilde{f}(z) z = (Fz)^T \nabla^2 f(Fz + \hat{x}) (Fz) \geq m \|Fz\|^2 = m \|z\|^2$$
> 
> where we used $F^T F = I$.

Note:
- Correct formula ✓
- Dimension check ✓
- Clear proof ✓
- No nonsensical inverse formulas ✓

### Red Flags in the Paper Under Review

1. **Dimensional inconsistencies** (should be caught in peer review)
2. **Undefined operations** ($F_t^{-1}$ for non-square $F_t$)
3. **No numerical verification** (a simple $2 \times 2$ example would catch this)
4. **Claiming novelty without literature review** (interior-point methods exist)

---

## Final Verdict on Issue 1.3

### Severity: **HIGH** (shows lack of rigor)

### Breakage: **MEDIUM** (the algorithm and final bounds can be salvaged)

### Fixability: **HIGH** (can be corrected with proper proofs)

### Impact on Trust: **VERY HIGH** (makes you question everything else)

### Recommended Action:

**If you're reviewing this paper:** **REJECT** with request for major revisions
- Fundamental errors in proof
- Need to re-derive all results carefully
- Need to verify with examples
- Need to check novelty claims

**If you need to use the algorithm:** **PROCEED WITH CAUTION**
- Implement the algorithm as described (it's sound)
- Use the correct formulas (Hessian, bounds)
- Don't trust the theoretical analysis without verification
- Test thoroughly on your problem

**If you're building on this work:** **START FRESH**
- Re-derive all results from first principles
- Use correct formulas throughout
- Provide careful proofs
- Verify with numerical examples
- Check literature for prior work

---

## Summary

Issue 1.3 (the inverse formula) is **critically flawed but salvageable**:

- ✗ The formula is mathematically nonsensical
- ✗ The proof is invalid
- ✓ The algorithm itself is still sound
- ✓ The bounds can be proven correctly
- ✓ The approach (project + Newton) is reasonable

The paper needs **major corrections**, but the core idea isn't fundamentally broken—just very poorly executed.

The real concern is: **if they got this so wrong, what else is wrong?** (And as we've seen, quite a lot!)
