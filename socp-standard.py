# Import packages.
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import qdldl
np.random.seed(2)
# Generate a random feasible SOCP.
m = 3
n = 10
p = 5
f = np.random.randn(n)
x0 = np.random.randn(n)
F = sp.random(p, n, density=0.2, format='csr')
g = F @ x0
x = cp.Variable(n)
# Define a single large sparse second-order cone constraint
A = sp.random(m * (n + 1), n, density=0.2, format='csr')
b = np.random.randn(m * (n + 1))
c = np.random.randn(n)
constraint = [cp.norm2(A @ x + b) <= c.T @ x]
# Define and solve the CVXPY problem.
prob = cp.Problem(cp.Minimize(f.T@x),[F @ x == g] + constraint)
data, _, _ = prob.get_problem_data(solver=cp.CLARABEL)
print(data)
# Print result.
#prob.solve(solver=cp.CLARABEL, verbose=True)
print(x.value)

F = qdldl.Solver(data['A'])
init_sol = F.solve(data['b'])
