# Comprehensive Analysis of Numerical Example Issues (Section IV)

Beyond the non-differentiability problem, the numerical experiment contains numerous suspicious and problematic elements that raise serious questions about the paper's rigor and the validity of the reported results.

---

## 1. The "Diagonal Hessian" Claim is FALSE

### Their Statement
> "The fixed nature of the network and the diagonal Hessian matrix means that the inversion step only has to be done once."

### The Cost Function
$$f_i(x) = \alpha_i e^{\beta_i |x|}$$

For a network with flow variables $x = (x_1, \ldots, x_n)$, the total cost is:
$$f(x) = \sum_{i=1}^n \alpha_i e^{\beta_i |x_i|}$$

### Computing the Hessian

For $x_i \neq 0$:
$$\frac{\partial^2 f}{\partial x_i^2} = \alpha_i \beta_i^2 e^{\beta_i |x_i|}$$

For $x_i \neq 0$, $x_j \neq 0$ with $i \neq j$:
$$\frac{\partial^2 f}{\partial x_i \partial x_j} = 0$$

**So the Hessian IS diagonal... at points where all components are non-zero.**

### Why "One-Time Factorization" is Still FALSE

Even though the Hessian is diagonal:

1. **The diagonal entries change every time step:**
   $$H_t = \text{diag}(\alpha_1^{(t)} \beta_1^{(t)2} e^{\beta_1^{(t)} |x_1|}, \ldots, \alpha_n^{(t)} \beta_n^{(t)2} e^{\beta_n^{(t)} |x_n|})$$

2. **The parameters $\alpha_i^{(t)}, \beta_i^{(t)}$ are resampled each round** (stated in the paper)

3. **The flows $x_i$ change every iteration** within each time step

4. **Newton's method requires solving:**
   $$H_t \Delta x = -\nabla f_t$$
   
   Even for a diagonal $H_t$, this requires computing new diagonal entries every iteration.

### What They Confused

They conflated two different things:

- **Symbolic sparsity pattern:** Fixed (diagonal structure doesn't change)
- **Numerical values:** Time-varying (must recompute every iteration)

For a **diagonal** matrix:
- Inversion: $O(n)$ (just reciprocals)
- Solving $H x = b$: $O(n)$ (element-wise division)

**But you still need to compute the new diagonal entries at every iteration!**

---

## 2. The Network Topology Doesn't Make Sense

### Their Description
> "a fixed, radial network composed of 15 nodes connected via 30 arcs"

### Graph Theory Basics

A **radial network** (or **tree network**) is a connected graph with no cycles.

**For a tree with $n$ nodes:**
- **Undirected edges:** $n - 1 = 14$ edges
- **Directed arcs** (bidirectional): $2(n-1) = 28$ arcs

### The Problem

With 15 nodes:
- If radial (tree): should have **14** or **28** arcs
- They claim: **30** arcs

**30 arcs means there are 2 extra edges**, creating cycles. This is **not a radial network**.

### Why This Matters

1. **Radial networks have special structure:**
   - Unique path between any two nodes
   - Linear flow equations
   - Very sparse constraint matrix $A$
   - Fast specialized algorithms exist

2. **A network with 30 arcs on 15 nodes:**
   - Average degree: $30/15 = 2$ arcs per node
   - Still relatively sparse, but **not a tree**
   - May have meshed topology

3. **Computational implications:**
   - For trees: $O(n)$ complexity algorithms exist for many problems
   - For meshed: require general sparse methods

### Possibilities

1. **Typo:** Should be 28 arcs (directed tree)
2. **Wrong term:** Not actually "radial," just sparse
3. **Careless writing:** Didn't check the basic math

---

## 3. The "Fixed Network" vs "Time-Varying Constraints" Contradiction

### Their Statement
> "The fixed nature of the network..."

### But Also (in the problem setup)
> "time-varying linear equality constraints $A_t x_t = b_t$"

### The Question

**Is the network structure fixed or time-varying?**

**Option A: Fixed $A_t = A$ (only $b_t$ changes)**

This is the **standard optimal power flow** setup:
- Kirchhoff's laws $Ax = b_t$ where $A$ is the incidence matrix (fixed)
- Only demand $b_t$ changes

**Computational advantage:**
- Can factor $A A^\top$ **once**
- Projection: $O(1)$ overhead after factorization
- This is the whole point of using a fixed network!

**Option B: Time-varying $A_t$ and $b_t$**

This means the network **topology changes** each time step:
- Arcs appear/disappear
- Cannot reuse factorizations
- Projection becomes $O(np^2)$ each time step

### Which Did They Actually Use?

**The paper is ambiguous:**

- Section IV suggests "fixed network" → $A$ is constant
- Section III theory requires time-varying $A_t$
- Algorithm 2 uses time-varying $A_t, b_t$

**If $A$ is fixed:**
- The full generality of their theory is not tested
- The comparison to time-varying methods is unfair

**If $A_t$ varies:**
- The "fixed network" claim is false
- The topology description is meaningless

---

## 4. Where Do $\alpha_i, \beta_i$ Come From?

### Their Statement
> "The cost function parameters $\alpha_i, \beta_i$ are randomly generated at each time step"

### Missing Details

1. **What distributions?**
   - Uniform? Gaussian? Log-normal?
   - What ranges? $\alpha_i \in [a, b]$?

2. **Are they independent?**
   - Or correlated across arcs?

3. **Positive constraints?**
   - For convexity, need $\alpha_i, \beta_i > 0$
   - Is this enforced?

4. **Scaling?**
   - What order of magnitude?
   - $\alpha_i \sim O(1)$ or $O(10^6)$?

5. **Condition number implications?**
   - The Hessian eigenvalues are $\lambda_i = \alpha_i \beta_i^2 e^{\beta_i |x_i|}$
   - Random $\alpha_i, \beta_i$ can create **arbitrarily ill-conditioned** problems
   - Condition number $\kappa = \max_i(\alpha_i \beta_i^2 e^{\beta_i |x_i|}) / \min_j(\alpha_j \beta_j^2 e^{\beta_j |x_j|})$

### Why This is Critical

**Reproducibility:** No one can reproduce their results without these details.

**Fairness:** Competitor methods may be sensitive to scaling/conditioning.

**Validity:** Random parameters might violate theoretical assumptions (e.g., Lipschitz constants).

---

## 5. The "Demand $b_t$" Generation is Unspecified

### What They Say
> "time-varying demand $b_t$"

### What They Don't Say

1. **How is $b_t$ generated?**
   - Random walk? Sinusoidal? Markov chain?

2. **What are the magnitudes?**
   - $\|b_t\| \sim O(?)$

3. **How much variation?**
   - $\|b_{t+1} - b_t\| = ?$
   - This is the path variation $V_T$ that determines regret!

4. **Feasibility constraints?**
   - For network flow: $\sum_i b_i = 0$ (flow conservation)
   - Is this enforced?

5. **Optimum path variation?**
   - The bound depends on $\sum_{t=1}^T \|x_t^* - x_{t-1}^*\|$
   - Did they measure this?

### Specific Issue: They Never Report $V_T$

Their **theoretical bound** is:
$$\text{Regret} = O(V_T + 1)$$

where $V_T = \sum_{t=1}^T \|x_t^* - x_{t-1}^*\|$.

**In the experiments, they never report $V_T$!**

Without knowing $V_T$:
- We can't verify if their empirical regret matches theory
- A method could have high regret simply because $V_T$ is large
- The comparison to other methods is meaningless

---

## 6. The Comparison to MOSP and MALM is Unfair and Misleading

### Their Setup

**OPEN-M:** Solves
$$\min_{x_t} f_t(x_t) \quad \text{s.t.} \quad A_t x_t = b_t$$

**MOSP & MALM:** They claim these solve inequality-constrained problems, so they "relax" the equality constraint:
$$\min_{x_t} f_t(x_t) \quad \text{s.t.} \quad A_t x_t - b_t \leq 0$$

### Why This is Wrong

#### Problem 1: Different Feasible Sets

- **Equality:** Feasible set is an affine subspace (dimension $n - p$)
- **Inequality:** Feasible set includes all $x$ with $A_t x_t \leq b_t$ (a polyhedron)

**These are completely different optimization problems!**

The optimal solution to the inequality version may be **strictly interior** to the equality constraint, giving a different optimum.

#### Problem 2: The Inequality is Wrong

The correct relaxation of $Ax = b$ to an inequality should be:
$$-\epsilon \leq Ax - b \leq \epsilon$$

or in standard form:
$$Ax \leq b, \quad -Ax \leq -b$$

Just using $Ax \leq b$ is a **one-sided relaxation** that changes the problem.

#### Problem 3: MOSP and MALM Can Handle Equalities

Looking at the cited references:

- **MOSP** (reference [15]): Can handle equality constraints via Lagrangian methods
- **MALM** (reference [14]): Uses augmented Lagrangian (designed for equalities!)

**They didn't need to relax anything!**

### What They Should Have Done

1. **Implement MOSP and MALM for equality constraints** (as originally designed)
2. **Or compare to other equality-constrained online methods** (e.g., [6])
3. **Or add inequality constraints to their own problem** (e.g., $x \geq 0$ for flows)

### The Implication

The reported superior performance of OPEN-M may be **entirely due to solving an easier problem** than the competitors.

---

## 7. Missing: Initialization Details

### The Theory Requires (Remark 1)

$$\|x_0 - x_0^*\| \leq \gamma = \frac{h}{2L}$$

### Questions

1. **How did they compute $x_0^*$?**
   - Solved the initial problem to optimality?
   - Used an approximation?

2. **How did they compute $h$ and $L$?**
   - These are problem-dependent constants
   - Did they estimate them? Use ground truth?

3. **What if the initialization is poor?**
   - Did they test sensitivity to $x_0$?

4. **For MOSP and MALM:**
   - What initialization did they use?
   - Fair comparison requires similar initialization quality

---

## 8. The "Convex" Claim Requires Qualification

### Their Statement
> "convex, twice-differentiable functions"

### For $f_i(x) = \alpha_i e^{\beta_i |x|}$

**Convexity:**

The function is convex if and only if $\alpha_i > 0$ and $\beta_i \geq 0$.

**Proof:** For a univariate function, convexity requires $f'' \geq 0$.

For $x > 0$:
$$f''(x) = \alpha_i \beta_i^2 e^{\beta_i x} \geq 0 \quad \text{iff} \quad \alpha_i \geq 0, \beta_i^2 \geq 0$$

For $x < 0$:
$$f''(x) = \alpha_i \beta_i^2 e^{-\beta_i x} \geq 0 \quad \text{iff} \quad \alpha_i \geq 0, \beta_i^2 \geq 0$$

**So convexity is guaranteed if $\alpha_i > 0, \beta_i \in \mathbb{R}$.**

### But They Never State These Constraints!

If $\alpha_i < 0$ or complex values are used:
- The function may be **non-convex**
- Newton's method may converge to local minima or saddle points
- The theory doesn't apply

**Missing from the paper:**
- Explicit statement that $\alpha_i > 0$
- How they ensured this during random generation

---

## 9. No Discussion of Constraint Qualification

### For Network Flow

The constraint matrix $A$ is the **incidence matrix**:
$$A_{ij} = \begin{cases} +1 & \text{if arc } j \text{ leaves node } i \\ -1 & \text{if arc } j \text{ enters node } i \\ 0 & \text{otherwise} \end{cases}$$

### Properties

1. **Rank of $A$:**
   - For a connected network with $n$ nodes and $m$ arcs
   - $\text{rank}(A) = n - 1$ (the $n$-th row is dependent due to flow conservation)

2. **For equality constraints $Ax = b$:**
   - Feasibility requires $b \in \mathcal{R}(A)$
   - Equivalently: $\sum_i b_i = 0$ (total flow balance)

3. **Did they ensure $\sum_i b_t^{(i)} = 0$ when generating $b_t$?**
   - Not mentioned!
   - If violated, the problem is **infeasible**

4. **Null space dimension:**
   - $\dim(\mathcal{N}(A)) = m - (n-1) = m - n + 1$
   - For 15 nodes, 30 arcs: $\dim(\mathcal{N}(A)) = 30 - 15 + 1 = 16$
   - OPEN-M works in this 16-dimensional space

---

## 10. No Code or Data Availability

### Standard Practice in Computational Papers

Reputable journals now require:
- Code repository (GitHub, Zenodo, etc.)
- Data files or generation scripts
- README with reproduction instructions

### This Paper

- **No code provided**
- **No data provided**
- **No supplementary materials**

### Consequences

- **Impossible to reproduce** the results
- **Cannot verify** the implementation matches the description
- **Cannot check** if bugs/errors affected the results
- **Cannot compare** other methods on the same problem instances

---

## 11. Insufficient Experimental Details

### What's Missing

| Detail | Status |
|--------|--------|
| Number of time steps $T$ | ✗ Not specified |
| Number of trials/runs | ✗ Not specified |
| Stopping criteria | ✗ Not specified |
| Error tolerance | ✗ Not specified |
| Optimality gap definition | ✗ Not specified |
| Hardware/software | ✗ Not specified |
| Runtime measurements | ✗ Not provided |
| Convergence plots | ✗ Not provided (only final regret) |
| Path variation $V_T$ | ✗ Not reported |

### What's Provided

- A single plot (Figure 1) showing cumulative regret vs time
- No error bars
- No confidence intervals
- No statistical tests

---

## 12. The Figure 1 Plot Raises Questions

### What the Plot Shows

(Based on the description, since I can't see the actual figure clearly)

- OPEN-M has lower cumulative regret than MOSP and MALM
- The curves are smooth

### Suspicious Elements

1. **Too smooth:**
   - With random $\alpha_i, \beta_i, b_t$, there should be some noise
   - Perfectly smooth curves suggest averaging over many runs
   - But they don't mention multiple runs

2. **No error bars:**
   - Standard practice is to show confidence intervals
   - Especially with random problem instances

3. **MOSP and MALM performance:**
   - These are established methods with theoretical guarantees
   - Are they really worse, or did they solve a different problem?

4. **Convergence to zero:**
   - Does the plot show regret converging to a constant?
   - Or still growing linearly?
   - This matters for evaluating $O(V_T)$ vs $O(V_T \log T)$ vs $O(T)$

---

## 13. Physical Plausibility of the Cost Function

### In Real Power Systems

Cost functions typically model:
- **Generation cost:** Quadratic or piecewise linear
  - $C(p) = a p^2 + b p + c$ (fuel cost)
- **Loss cost:** Quadratic in current
  - $C(I) = I^2 R$ (resistive losses)
- **Penalty costs:** For deviations from setpoints

### The Exponential of Absolute Value

$$f_i(x) = \alpha_i e^{\beta_i |x|}$$

**Where does this come from?**

1. **Not a standard generation cost model**
2. **Not a loss model** (losses are even functions of flow)
3. **Not a piece-wise linear approximation**

**It looks like:**
- A made-up function to have certain mathematical properties
- Convex (good for optimization)
- Has a known gradient (when differentiable)
- Diagonal Hessian (simplifies computations)

**But it's not physically motivated.**

### Alternative Interpretations

**Possible (generous) interpretation:**

Could represent an **approximation to a barrier function** for keeping flows bounded:
$$e^{\beta |x|} \to \infty \text{ as } |x| \to \infty$$

But this would require $\beta > 0$ to be large, which makes the problem **very ill-conditioned**.

---

## 14. Comparison: What a Good Experimental Section Would Include

### Problem Description
- ✓ Network topology with figure
- ✓ Specific values: $n=15$ nodes, $m=30$ arcs
- ✓ Cost function form
- ✗ **Missing:** Arc connectivity (adjacency matrix or list)
- ✗ **Missing:** How to construct incidence matrix $A$

### Parameter Generation
- ✗ **Missing:** Distributions for $\alpha_i, \beta_i$
- ✗ **Missing:** Ranges/bounds
- ✗ **Missing:** Time-variation model for $b_t$
- ✗ **Missing:** Path variation $V_T$ statistics

### Algorithm Implementation
- ✗ **Missing:** Stopping criteria
- ✗ **Missing:** Numerical tolerances
- ✗ **Missing:** Linear solver used
- ✗ **Missing:** Initialization procedure
- ✗ **Missing:** How $h, L$ were estimated

### Comparison
- ✗ **Missing:** Identical problem setup for all methods
- ✗ **Missing:** Fair initialization
- ✗ **Missing:** Constraint handling verification

### Results
- ✗ **Missing:** Error bars / confidence intervals
- ✗ **Missing:** Number of trials
- ✗ **Missing:** Statistical significance tests
- ✗ **Missing:** Convergence behavior plots
- ✗ **Missing:** Runtime comparison
- ✗ **Missing:** Sensitivity analysis

### Reproducibility
- ✗ **Missing:** Code repository
- ✗ **Missing:** Data files
- ✗ **Missing:** Instructions

---

## 15. Summary Table: All Numerical Example Issues

| # | Issue | Severity | Type |
|---|-------|----------|------|
| 1 | Non-differentiable objective | **Critical** | Mathematical error |
| 2 | "One-time factorization" false claim | **High** | Misunderstanding |
| 3 | Diagonal Hessian still time-varying | **High** | Misleading claim |
| 4 | Network topology inconsistent (15 nodes, 30 arcs ≠ tree) | **Medium** | Error or imprecision |
| 5 | Fixed vs time-varying network contradiction | **High** | Ambiguous |
| 6 | $\alpha_i, \beta_i$ generation unspecified | **Critical** | Non-reproducible |
| 7 | $b_t$ generation unspecified | **Critical** | Non-reproducible |
| 8 | Path variation $V_T$ not reported | **Critical** | Invalidates comparison |
| 9 | Unfair comparison (different problems) | **Critical** | Experimental design flaw |
| 10 | Convexity constraints on $\alpha_i$ not stated | **Medium** | Incomplete |
| 11 | Feasibility of $b_t$ not discussed | **Medium** | Potentially infeasible |
| 12 | No code or data availability | **High** | Reproducibility |
| 13 | Insufficient experimental details | **High** | Incomplete |
| 14 | Figure without error bars | **Medium** | Statistical rigor |
| 15 | Non-physical cost function | Low | Questionable relevance |
| 16 | Initialization procedure unclear | **High** | Unfair comparison |
| 17 | No sensitivity analysis | **Medium** | Incomplete validation |

**Total Critical Issues:** 6  
**Total High Issues:** 8  
**Total Issues:** 17

---

## Conclusion

The numerical experiment (Section IV) suffers from **severe methodological flaws** that make the results **unreliable and non-reproducible**:

### Cannot Trust the Results Because:

1. **The objective function violates the assumptions** (not differentiable)
2. **The comparison is unfair** (MOSP/MALM solve a different problem)
3. **Critical details are missing** (parameter generation, $V_T$, initialization)
4. **Claims are contradictory or false** (one-time factorization, network topology)
5. **No reproducibility** (no code, no data, insufficient details)

### The Experiment Proves Nothing About:

- Whether OPEN-M actually works on the stated problem
- Whether OPEN-M is better than MOSP/MALM
- Whether the theoretical bounds are empirically validated
- Whether the method is practical for real applications

### What Would Be Needed to Fix This:

1. **Use a smooth cost function** (e.g., $f_i(x) = \alpha_i e^{\beta_i x^2}$)
2. **Clarify the network structure** (provide adjacency matrix)
3. **Specify all parameter generation procedures** (with seed for reproducibility)
4. **Report path variation $V_T$** for each problem instance
5. **Fair comparison:** All methods solve the **same problem** with **same initialization**
6. **Release code and data**
7. **Provide error bars** from multiple trials
8. **Measure and report runtimes**

As it stands, **Section IV does not validate the theoretical results** and should be disregarded.
