# Initialization Analysis for Online Interior Point Methods

Based on Section 3 of the paper and practical considerations for online IPM implementation.

## The Initialization Challenge

The paper states:
> "Suppose we have an initial primal-dual point x₀, ν₀ and an initial barrier parameter η₀ such that ‖n₀(y₀, η₀)‖_{D(y₀)} ≤ 1/9 and Ax₀ = b₀. In practice, this can be achieved using offline optimization before the start of the online procedure."

This raises several practical questions about initialization:

## 1. Finding the Initial Barrier Parameter η₀

### What is the Barrier Parameter?

The barrier parameter η (often called μ in standard IPM literature) controls the central path trajectory. In barrier methods:
- The central path is defined as the set of points (x(η), ν(η)) that satisfy the perturbed KKT conditions
- As η → 0, the central path approaches the optimal solution
- Larger η values keep iterates away from the constraint boundaries

### Central Path and Initialization

**Key Insight**: You don't necessarily need to start exactly ON the central path, but you need to be in a neighborhood of it.

The condition ‖n₀(y₀, η₀)‖_{D(y₀)} ≤ 1/9 means:
- n₀(y₀, η₀) is the Newton step (or residual) at the initial point
- D(y₀) is some norm (likely related to the Hessian)
- The bound 1/9 ensures we're "close enough" to the central path

### Practical Approaches to Find η₀:

#### Option 1: Adaptive η₀ Selection
```
1. Start with a feasible point (x₀, ν₀) where Ax₀ = b₀
2. Try different values of η₀ (e.g., η₀ ∈ {1, 0.1, 0.01, 0.001})
3. For each η₀, compute ‖n₀(y₀, η₀)‖_{D(y₀)}
4. Choose the largest η₀ that satisfies ‖n₀(y₀, η₀)‖_{D(y₀)} ≤ 1/9
```

#### Option 2: Heuristic η₀ Based on Problem Scale
```
η₀ = α · ‖c‖ / n
```
Where α ∈ [0.1, 1] is a scaling factor, c is the objective vector, and n is the problem dimension.

#### Option 3: Complementarity-Based η₀
For problems with inequality constraints (if applicable):
```
η₀ = (x^T s) / m
```
Where s are slack variables and m is the number of constraints.

## 2. Finding a Feasible Initial Point Ax₀ = b₀

This is a classic "Phase I" problem in optimization. Several approaches:

### Option 1: Artificial Variables Method
```
Solve the auxiliary problem:
minimize    ‖w‖₁
subject to  Ax + w = b₀
           x ≥ 0 (if applicable)

If ‖w*‖₁ = 0, then x* is feasible for the original problem.
Set x₀ = x*.
```

### Option 2: Least Squares Projection
```
x₀ = argmin ‖Ax - b₀‖²

This gives: x₀ = A^†b₀ (Moore-Penrose pseudoinverse)

Note: This only gives Ax₀ = b₀ if b₀ ∈ range(A)
```

### Option 3: Big-M Method
```
Introduce artificial variables and solve:
minimize    M·(sum of artificial variables)
subject to  Ax + artificial variables = b₀
           x ≥ 0, artificial variables ≥ 0

Where M is a large positive constant.
```

### Option 4: Feasible Point from Previous Time Step
```
In the online setting, use the solution from the previous time period:
x₀ = x_{t-1}*

This may violate Ax₀ = b₀ if constraints changed, requiring adjustment.
```

## 3. Practical Implementation Strategy

### Complete Initialization Algorithm:
```python
def initialize_online_ipm(A, b0, c):
    # Step 1: Find initial feasible point
    x0 = find_feasible_point(A, b0)
    
    # Step 2: Set initial dual variables (heuristic)
    nu0 = np.linalg.lstsq(A.T, c, rcond=None)[0]
    
    # Step 3: Find appropriate barrier parameter
    eta0 = find_barrier_parameter(x0, nu0, A, b0)
    
    # Step 4: Verify initialization condition
    assert check_initialization_condition(x0, nu0, eta0) <= 1/9
    
    return x0, nu0, eta0

def find_feasible_point(A, b0):
    """Find x0 such that Ax0 = b0 using least squares"""
    return np.linalg.pinv(A) @ b0

def find_barrier_parameter(x0, nu0, A, b0):
    """Find largest eta0 satisfying the initialization condition"""
    eta_candidates = [1.0, 0.1, 0.01, 0.001, 0.0001]
    
    for eta in eta_candidates:
        if check_initialization_condition(x0, nu0, eta) <= 1/9:
            return eta
    
    # If none work, use smallest value and hope for the best
    return eta_candidates[-1]
```

## 4. Alternative: Warm Start Strategy

For online problems, a practical approach is:

### Rolling Initialization:
```python
# At time t:
if t == 0:
    # Cold start: use methods above
    x0, nu0, eta0 = full_initialization(A0, b0, c0)
else:
    # Warm start: use previous solution
    x0 = x_prev
    nu0 = nu_prev
    
    # Adjust for constraint changes
    if not np.allclose(A @ x0, b0):
        x0 = project_to_feasible(x0, A, b0)
    
    # Re-scale barrier parameter
    eta0 = min(eta_prev, find_barrier_parameter(x0, nu0, A, b0))
```

## 5. Key Practical Considerations

1. **Constraint Changes**: In online settings, both A and b change over time
2. **Computational Budget**: Initialization shouldn't take too long
3. **Numerical Stability**: Ensure initialization doesn't start too close to boundaries
4. **Problem Structure**: Exploit any special structure (LP, QP, etc.)

## 6. Recommendation

For practical implementation:
1. Use least squares projection for initial feasible point
2. Use heuristic dual variable initialization
3. Adaptively select barrier parameter
4. Implement warm start for t > 0
5. Always verify the initialization condition numerically

This approach balances theoretical requirements with computational practicality.