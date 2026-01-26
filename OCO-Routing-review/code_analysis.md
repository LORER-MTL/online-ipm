# Code Analysis: OlivierBelanger/oop_opt Implementation

**Date:** January 2026

---

## Executive Summary

The implementation code reveals significant **discrepancies between the paper's claims and the actual implementation**, as well as concerning practices around data regeneration and algorithm implementation.

---

## 1. Parameter Discrepancies

### Paper vs. Code Comparison

| Parameter | Paper Claims | Code (`constants.py`) | Discrepancy |
|-----------|--------------|----------------------|-------------|
| NUM_SIM | 100 Monte Carlo runs | **1** | 99x fewer runs |
| P (priorities) | 3 | **2** | Different problem size |
| M (modem banks) | 16 | **2** | 8x smaller |
| λ rates | {20, 25, 30} | **{10, 30, 50}** | Different values |
| k_p (costs) | {10, 4, 1} or {4, 2, 1} | **{4, 1}** | Different costs |

**Impact:** The code tests a problem with:
- 2×2 = 4 queues instead of 3×16 = 48 queues
- N = 144 variables instead of N = 1728 variables
- Much smaller and potentially easier problem

---

## 2. The `--reuse_data` Flag

### What It Does

From `main.py`:
```python
parser.add_argument('--reuse_data', action='store_true', help='Reuse old data for plotting')
parser.add_argument('--data_file', type=str, default='sim_data_20240819_131147.dat', ...)
```

From `utils.py`:
```python
def run_scenario_comp(reuse_data=False, data_file='test.dat'):
    if reuse_data:
        full_data_file = 'oop_opt/results/' + data_file
        loaded_data = load_data_from_file(full_data_file)
        # ... just plot from saved data, don't run simulation
```

### Why This Is Suspicious

1. **Cherry-picking potential:** Can skip simulation and plot from any saved file
2. **Default file is timestamped:** `sim_data_20240819_131147.dat` - a specific run from August 2024
3. **No reproducibility guarantee:** Results depend on which saved file you use
4. **Opposite of scientific practice:** Should regenerate data to verify results

### Is Regenerating Data Useless?

**No, regenerating is essential!** The `--reuse_data` flag is the suspicious part because:
- Regenerating data with the same parameters should give statistically similar results
- If results aren't reproducible, there's a problem with the algorithm or parameters
- The fact they provide a flag to skip regeneration suggests they may know the results vary

---

## 3. Critical Algorithm Difference: IPM vs IPMFeasible

### Two Classes, Different Behavior

**`IPM` class (Algorithm.py:190-241)** - What they actually use:
```python
def update(self, prob):
    delta = sp.linalg.spsolve(D, augGrad)
    norm = grad.T.dot(delta[:self.n])
    if norm > 1 and self.damped:
        delta = delta/norm  # Weak damping, NOT feasibility check
    self.x -= delta[:self.n]  # Update WITHOUT checking feasibility
```

**`IPMFeasible` class (Algorithm.py:273-331)** - With line search:
```python
def update2(self, prob):
    delta = sp.linalg.spsolve(D, augGrad)
    while(not prob.barr.isFeasible(self.x - delta[:self.n])):
        delta *= 0.8  # Backtrack until feasible
        count += 1
    self.x -= delta[:self.n]
```

### Which One Do They Use?

From `mpc_problem.py:261`:
```python
oipm_tec = IPM(n, SIZE_b)  # NOT IPMFeasible!
```

**They use `IPM`, NOT `IPMFeasible`!**

This means:
- **No line search for feasibility**
- Only weak gradient-based damping (if norm > 1, divide by norm)
- Newton step can exit the feasible region
- Barrier function can become undefined

---

## 4. The Feedback Correction Implementation

### What It Actually Does

From `mpc_problem.py:273-285`:
```python
for p in range(P):
    optimal_f_in_row_sum = row_sums[p]
    if optimal_f_in_row_sum != 0:
        optimal_f_in_corrected[p, :] = (self.flows[t][p] * optimal_f_in_reshaped[p, :]) / optimal_f_in_row_sum
    else:
        optimal_f_in_corrected[p, :] = np.zeros(M)
```

### Interpretation

This is **proportional scaling**:
```
corrected_f_in[p, m] = actual_demand[p] × (allocated_f_in[p, m] / total_allocated[p])
```

It redistributes the inflow allocations to match actual demand while preserving the relative allocation across modem banks.

### What It Does NOT Do

- Does NOT fix Newton steps that exit the feasible region
- Does NOT handle inequality constraint violations (queue capacity, bandwidth)
- Only fixes the demand-matching equality constraint (1d)

---

## 5. Initialization Procedure

### How They Find Initial x_0

From `mpc_problem.py:224-239`:
```python
if start_t == 0:
    while True:
        self.x_start = np.random.rand(n, 1)
        if oco_problem.barr.isFeasible(self.x_start.transpose()):
            break

    initial_guess_alg = IPMFeasible(n, SIZE_b)  # Note: IPMFeasible here!
    initial_guess_alg.setX(self.x_start.transpose())

    for i in range(50):
        self.x_start = initial_guess_alg.etaUpdateLimit(oco_problem)
```

### Analysis

1. **Random sampling until feasible:** Could take many tries
2. **50 iterations of `IPMFeasible` to warm-start:** This DOES use line search
3. **But main loop uses `IPM` without line search:** Inconsistent

So they warm-start carefully with line search, but then run the main algorithm without it!

---

## 6. Barrier Gradient Blowup

### Empirical Test Results

Testing `LinLogBar` barrier with constraint `x[0] <= 1`:

| Distance to Boundary (slack) | |Gradient| |
|------------------------------|-----------|
| 0.5 (50%) | 2 |
| 0.01 (1%) | 100 |
| 0.001 (0.1%) | 1000 |

**Conclusion:** Gradient is O(1/slack), confirming that near the boundary:
- Gradients become huge
- Newton steps become large
- Risk of exiting feasible region increases

---

## 7. Summary of Issues

### Theoretical Issues (from paper analysis)
- OIPM-TEC proofs are invalid
- No line search in algorithm
- Barrier Hessian ill-conditioning

### Implementation Issues (from code analysis)
- Parameters don't match paper (P=2 vs 3, M=2 vs 16, NUM_SIM=1 vs 100)
- `--reuse_data` flag allows skipping simulation
- Uses `IPM` (no line search) instead of `IPMFeasible` (has line search)
- Warm-start uses different algorithm than main loop

### What This Means

1. **Results in paper may not be reproducible** with this code
2. **The code tests a much smaller problem** than claimed
3. **The algorithm has weaker guarantees** than `IPMFeasible` would provide
4. **The ability to skip regeneration is suspicious** in a research context

---

## 8. NumPy Compatibility Issues

The code has a compatibility issue with modern NumPy (2.x):

**Line 215 in Algorithm.py:**
```python
viol.data = np.round(viol.data, 8)
```

The `.data` attribute is for sparse arrays, but the code sometimes produces dense arrays. This causes:
```
AttributeError: attribute 'data' of 'numpy.ndarray' objects is not writable
```

**Impact:** Cannot reproduce results with NumPy 2.x without code modifications.

---

## 9. Reproduction Results (January 2026)

### Setup

After fixing several code issues (circular imports, missing `time` import, solver configuration, NumPy 1.x compatibility), simulations were run at multiple scales.

**Environment:**
- NumPy 1.26.4 (pinned due to compatibility issues)
- CLARABEL solver
- T=100, W=5, NUM_SIM=1

### Scaling Results

| Config | Queues | Offline Loss | OCMPC % Worse | MPC % Worse | OCMPC/MPC Ratio | Newton Norms |
|--------|--------|--------------|---------------|-------------|-----------------|--------------|
| P=2, M=2 | 4 | 1217 | **14.5%** | 1.9% | **7.7x** | 113-175 |
| P=3, M=4 | 12 | 3646 | **20.2%** | 0.7% | **28x** | 513-579 |
| P=3, M=8 | 24 | 7178 | **24.9%** | 2.4% | **10x** | 1020-1027 |
| P=3, M=16 | 48 | — | **STUCK** | — | — | — |

### Critical Finding: OCMPC Consistently Underperforms MPC

**The online IPM algorithm (OCMPC/εOIPM-TEC) is 10-28x worse than standard MPC relative to optimal across all tested scales.**

### Paper Parameters (P=3, M=16) Cannot Run

With the paper's claimed parameters (P=3, M=16 = 48 queues, n=1728 variables):
- Initialization **gets stuck indefinitely** in the random feasibility search
- The code samples `x = np.random.rand(n, 1)` until `barr.isFeasible(x)` returns True
- In n=1728 dimensions, random sampling almost never hits the feasible polytope
- **The paper's results cannot be reproduced with their own code**

### Newton Step Norm Scaling

Newton norms scale roughly linearly with problem size:
- P=2, M=2: 113-175 (avg ~140)
- P=3, M=4: 513-579 (avg ~546)
- P=3, M=8: 1020-1027 (avg ~1024)

Large norms indicate operation near constraint boundaries. When `norm > 1`, the code divides by norm (weak damping), but this doesn't guarantee feasibility.

### Code Fixes Required to Run

1. **Circular import:** `utils/utils.py` imported itself - removed line 9
2. **Missing import:** Added `import time` to `utils/utils.py`
3. **Solver:** Changed from MOSEK to CLARABEL in `solver.py`
4. **Results directory:** Created `OlivierBelanger/results/`
5. **NumPy version:** Pinned to `numpy<2.0` in `pyproject.toml`
6. **Package naming:** Created symlink `oop_opt -> OlivierBelanger`
7. **Missing __init__.py:** Created in `utils/` directory

---

## 10. Recommendations

1. **Run with regenerated data:** Don't trust `--reuse_data`
2. **Match paper parameters:** Set P=3, M=16, NUM_SIM=100
3. **Use `IPMFeasible` instead of `IPM`** for proper line search
4. **Report constraint violations:** Add metrics for inequality constraint satisfaction
5. **Stress test:** Try higher traffic loads to see when the algorithm fails
6. **Fix NumPy compatibility:** Modify line 215 to handle dense arrays
7. **Investigate OCMPC underperformance:** Why does the online IPM perform worse than standard MPC?
