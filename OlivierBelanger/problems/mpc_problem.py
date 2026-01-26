import cvxpy as cp
import numpy as np
from .base_problem import BaseProblem
from ..constants import *
from ..solvers.solver import Solver
from ..simulators.simulator import Simulator
from ..mat_gen.matrices_gen import *
import time

from ..ocotools.Problem import *
from ..ocotools.Algorithm import *


class MPCProblem(BaseProblem):
    def __init__(self, name, start_t, end_t, flows, realized_flows, W, strategy="mpc", problem_type = "eq"):
        """Initialize an MPCProblem with specified parameters."""
        super().__init__(name, start_t, end_t)
        self.flows = flows
        self.realized_flows = realized_flows
        self.strategy = strategy
        self.W = W
        self.problem_type = problem_type

        self.initialize_decision_variables(start_t, end_t)

        self.f_in_accumulator = []
        self.f_out_accumulator = []
        self.L_accumulator = []
        self.L_oco_accumulator = []
        self.Q_accumulator = []
        self.delta_Q_accumulator = []
        self.Q_solver_accumulator = []
        self.w_accumulator = []

        self.Q = np.zeros((T+1, P, M))
        self.w = {t: cp.Variable((P, M), name = "w", nonneg=True) for t in range(T)}
        self.prev_w = {t: cp.Variable((P, M), name="w", nonneg=True) if self.problem_type == "eq" else np.zeros((P, M)) for t in range(T)}

        self.optim_w_start_0 = np.zeros((P, M))
        self.x = cp.Variable((M*P*NUM_VAR_TYPES* (self.W+1), 1))
        self.x.value = np.zeros((M*P*NUM_VAR_TYPES* (self.W+1), 1))
        self.prev_x_rolled = np.zeros((M*P*NUM_VAR_TYPES* (self.W+1), 1))

        self.prev_b = np.zeros((SIZE_b, 1))
        self.C_matrix_concat = None
        self.d_vector_concat = None
    
    
    def initialize_decision_variables(self, start_t, end_t):
        """Initializes decision variables for the MPC problem."""
        
        if self.problem_type == "mat":
            num_elem_per__var = M * P
            self.x = cp.Variable((M*P*NUM_VAR_TYPES* (self.W+1), 1))
            self.prev_x_rolled = np.zeros((M*P*NUM_VAR_TYPES* (self.W+1), 1))
            self.x.value = np.zeros((M*P*NUM_VAR_TYPES* (self.W+1), 1))

            self.f_in           = cp.Variable(num_elem_per__var * (self.W+1), name="f", nonneg=True)
            self.w              = cp.Variable(num_elem_per__var * (self.W+1), name="w", nonneg=True)
            self.L_solver       = cp.Variable(num_elem_per__var * (self.W+1), name="L_solver", nonneg=True)
            self.f_out_solver   = cp.Variable(num_elem_per__var * (self.W+1), name="f_out_solver", nonneg=True)
            self.Q_solver       = cp.Variable(num_elem_per__var * (self.W+1), name="Q_solver", nonneg=True)
            self.delta_Q_solver = cp.Variable(num_elem_per__var * (self.W+1), name="delta_Q_solver")

            segments = []
            for i in range(self.W+1):
                start_idx = i * num_elem_per__var
                end_idx = (i + 1) * num_elem_per__var

                segment = cp.vstack([
                    cp.reshape(self.f_in[start_idx:end_idx], (num_elem_per__var, 1)),
                    cp.reshape(self.w[start_idx:end_idx], (num_elem_per__var, 1)),
                    cp.reshape(self.L_solver[start_idx:end_idx], (num_elem_per__var, 1)),
                    cp.reshape(self.f_out_solver[start_idx:end_idx], (num_elem_per__var, 1)),
                    cp.reshape(self.Q_solver[start_idx:end_idx], (num_elem_per__var, 1)),
                    cp.reshape(self.delta_Q_solver[start_idx:end_idx], (num_elem_per__var, 1))
                ])
                segments.append(segment)

            self.x = cp.vstack(segments)
            self.f_in_values = {0: {t: np.zeros((P, M)) for t in range(T)}}


        else:
            cvxpy_variables = {
                "f_out_solver": (T, (P, M), True),
                "Q_solver": (T, (P, M), True),
                "delta_Q_solver": (T, (P, M), False),  # can be negative (Q_in - Q_out)
                "L_solver": (T, (P, M), True),
            }
            np_zeros = {
                "f_in_values": (T, (P, M)),
                "L": (T, (P, M)),
            }

            for var, (range_end, shape, nonneg) in cvxpy_variables.items():
                setattr(self, var, self.initialize_variable_for_range(range_end, shape, name=var, nonneg=nonneg))

            for np_var, shape in np_zeros.items():
                setattr(self, np_var, self.initialize_np_zeros_for_range(*shape))
        
    @staticmethod
    def initialize_variable_for_range(range_end, shape, name, nonneg=True):
        return {t: cp.Variable(shape, name=name, nonneg=nonneg) for t in range(range_end)}
    
    @staticmethod
    def initialize_np_zeros_for_range(range_end, shape):
        return {t: np.zeros(shape) for t in range(range_end)}
    
    def create_objective(self, start_t, end_t):
        if self.problem_type == "mat":
            c = generate_c_vector(start_t)
            return cp.Minimize(c @ self.x)
        elif self.problem_type == "oco":
            return generate_c_vector(start_t)
        else: 
            return cp.Minimize(cp.sum([cp.sum(cp.multiply(cp.sum(self.L_solver[t], axis=1), k_p)) for t in range(start_t, end_t)]))


    def generate_matrices(self, start_t, end_t, Q_sim = None, w_t_minus_1_realised = None, optim_w_start_0 = None):

        A_matrix_concat = None
        b_vector_concat = None
        C_matrix_concat = None
        d_vector_concat = None
        
        for idx, t in enumerate(range(start_t, end_t)):
            A_matrix = generate_A_matrix(start_t, end_t, t, idx)
            b_vec_Q = Q_sim if t == start_t else 0
            b_vec_w_t_minus_1 = w_t_minus_1_realised if t == start_t else 0
            b_vector = generate_b_vector(t, start_t, self.flows[t, :], b_vec_Q, b_vec_w_t_minus_1, optim_w_start_0)

            g_c = generate_g_c_matrices(idx)
            concatenated_g_c_i = np.vstack(g_c)

            g_d_matrices = generate_g_d_matrices()
            concatenated_g_d_i = np.vstack(g_d_matrices)
            
            A_matrix_concat = A_matrix if A_matrix_concat is None else np.vstack((A_matrix_concat, A_matrix))
            b_vector_concat = b_vector if b_vector_concat is None else np.vstack((b_vector_concat, b_vector))
            C_matrix_concat = concatenated_g_c_i if C_matrix_concat is None else np.vstack((C_matrix_concat, concatenated_g_c_i))
            d_vector_concat = concatenated_g_d_i if d_vector_concat is None else np.vstack((d_vector_concat, concatenated_g_d_i))
            
        return A_matrix_concat, b_vector_concat, C_matrix_concat, d_vector_concat
    
    
    def create_constraints(self, start_t, end_t, F, Q_sim = None, w_t_minus_1_realised = None):
        
        constraints = []

        if self.problem_type == "mat":
            A_matrix_concat, b_vector_concat, C_matrix_concat, d_vector_concat = self.generate_matrices(start_t, end_t, Q_sim, w_t_minus_1_realised, self.optim_w_start_0)
            constraints.append(A_matrix_concat @ self.x == b_vector_concat)
            constraints.append(C_matrix_concat @ self.x <= d_vector_concat)

        else:
            if start_t == 0: #If window includes first time step
                constraints.append(self.Q_solver[0][:, :] == Q_0)
            
            for t in range(start_t, end_t):
                constraints.extend([cp.sum(self.w[t][:P, m]) == 1 for m in range(M)]) # Constraint 2
                constraints.extend([cp.sum(self.Q_solver[t][:P, m] + self.delta_Q_solver[t][:P, m]) <= Q_BAR for m in range(M)])
                constraints.extend([cp.sum(self.f_in[t][p, :M]) == F[t, p] for p in range(P)])

                for p in range(P):
                    for m in range(M):
                        constraints.append(-self.w[t][p, m]     <= 0)
                        constraints.append(self.w[t][p, m] -1   <= 0)
                        constraints.append(DELTA_S * self.f_out_solver[t][p, m] - self.w[t][p, m] <= 0)
                        constraints.append(cp.sum(self.f_in[t][p, m] - self.f_out_solver[t][p, m] - self.delta_Q_solver[t][p, m]) == self.L_solver[t][p,m])
                        
                        if end_t >= T: #If window includes last time step
                            constraints.append(self.Q_solver[end_t-1][p, m] == Q_0)
                            constraints.append(self.delta_Q_solver[end_t-1][p, m] == Q_0) # Constraint 7
                        
                        if t < end_t-1:  # Constraint 6
                            constraints.append(self.Q_solver[t+1][p, m] == self.Q_solver[t][p, m] + self.delta_Q_solver[t][p, m])
                            constraints.append(self.prev_w[t+1] == self.w[t])
                        if start_t > 0:
                            constraints.append(cp.abs(self.w[t][p, m] - self.prev_w[t][p, m]) <= DELTA_W_BAR)
                        constraints.append(self.Q_solver[start_t][p, m] == self.Q[start_t][p, m])
                    
        return constraints

    def initialize_problem(self, start_t =0, end_t =T):
        """Initialize the MPC problem with objective and constraints."""

        self.initialize_decision_variables(start_t, end_t)
        if self.problem_type == "mat":
            prev_w_mat = self.prev_w[start_t] if start_t > 0 else 0
            constraints = self.create_constraints(start_t, end_t, self.flows, self.Q[start_t], prev_w_mat)
        else:
            constraints = self.create_constraints(start_t, end_t, self.flows)
        
        objective = self.create_objective(start_t, end_t)

        return objective, constraints
    

    def solve(self, solver, start_t, end_t, oipm_tec = None):
        """ Solve the MPC problem using a given solver."""
        if self.problem_type == "mat":
            objective, constraints = self.initialize_problem(start_t =start_t, end_t =end_t)
            solver.solve_problem(objective,constraints)
            
            return self.get_x_slices(var_name = "w", window_index = 1), self.get_x_slices(var_name = "f_in", window_index = 1)
        elif self.problem_type == "eq":
            objective, constraints = self.initialize_problem(start_t =start_t, end_t =end_t)
            solver.solve_problem(objective,constraints)
            if start_t == 0:
                self.optim_w_start_0 = self.w[start_t].value
            return self.w[start_t].value, self.f_in[start_t].value
        
        elif self.problem_type ==  "oco":
            print("start_t = ", start_t)
            self.initialize_decision_variables(start_t, end_t)
            self.prev_w_mat = 0 if start_t == 0 else self.prev_w_mat

            A, b, C, d = self.generate_matrices(start_t, end_t, self.Q[start_t], self.prev_w_mat)
            c = self.create_objective(start_t, end_t)
            n = self.x.shape[0]
            oco_problem = OCOMPC(n, A, b.transpose(), C, d.transpose(), c)

            if start_t == 0:
                start_time = time.time()
                while True:
                    self.x_start = np.random.rand(n, 1)
                    if oco_problem.barr.isFeasible(self.x_start.transpose()):
                        break
                
                initial_guess_alg = IPMFeasible(n, SIZE_b)
                initial_guess_alg.setX(self.x_start.transpose())

                for i in range(50):
                    print("i = ", i)
                    self.x_start  = initial_guess_alg.etaUpdateLimit(oco_problem)
                oipm_tec.setX(self.x_start)
                end_time = time.time()
                print("Time elapsed for initial guess = ", end_time - start_time)
            else:
                oipm_tec.setX(self.prev_x_rolled)

            x_vec = oipm_tec.update(oco_problem)

            prev_x = x_vec
            shift_units = M * P * NUM_VAR_TYPES
            self.prev_x_rolled = np.roll(prev_x, -shift_units, axis=1)
            self.prev_x_rolled[:, -shift_units:] = prev_x[:, -shift_units:].copy()

            self.prev_w_mat = self.get_x_slices(var_name = "w", window_index = 1, x_vec = x_vec).value
            
            return self.get_x_slices(var_name = "w", window_index = 1, x_vec = x_vec), self.get_x_slices(var_name = "f_in", window_index = 1, x_vec = x_vec)


    def manage(self, batch_weights = None, batch_f_in = None):
        """Manage the MPC problem based on the defined strategy."""
        simulator = Simulator()
        solver = Solver()
        optimal_w = []
        n = self.x.shape[0]
        oipm_tec = IPM(n, SIZE_b)

        for t in range(T):
            horizon_end = min(T, t + self.W) if self.problem_type == "eq" else t + self.W

            if self.problem_type == "oco":
                
                optimal_w, optimal_f_in = self.solve(solver, start_t=t, end_t=horizon_end, oipm_tec = oipm_tec)
      
                optimal_f_in_val = optimal_f_in.value
                optimal_f_in_reshaped = optimal_f_in_val.flatten().reshape((P, M))
                row_sums = optimal_f_in_reshaped.sum(axis=1)
                optimal_f_in_corrected = np.copy(optimal_f_in_reshaped)

                for p in range(P):
                    optimal_f_in_row_sum = row_sums[p]
                    if optimal_f_in_row_sum != 0:
                        optimal_f_in_corrected[p, :] = (self.flows[t][p] * optimal_f_in_reshaped[p, :]) / optimal_f_in_row_sum
                    else:
                        optimal_f_in_corrected[p, :] = np.zeros(M)

                optimal_f_in_corrected_unflattened = optimal_f_in_corrected.flatten().reshape((P * M, 1))
                corrected_optimal_f_in_cvx = cp.Variable((P * M, 1))
                corrected_optimal_f_in_cvx.value = optimal_f_in_corrected_unflattened
                optimal_f_in = corrected_optimal_f_in_cvx

            else:
                optimal_w, optimal_f_in = self.solve(solver, start_t=t, end_t=horizon_end)

            if t == 0:
                optim_static_weights = optimal_w
            if self.strategy == "batch_mean":
                optimal_w = batch_weights[t].value
                optimal_f_in = batch_f_in[t].value
            if self.strategy == "static_batch_mean":
                optimal_w = optim_static_weights

            Q_curr = self.Q[t]
            self.f_out[t], self.Q[t+1], self.L[t], self.delta_Q[t], self.f_in_values[t] = simulator.run_one_time_step(optimal_w, optimal_f_in, Q_curr, self.flows, self.realized_flows,t, problem_type = self.problem_type)
            
            if t < T-1 and self.problem_type == "eq":
                self.Q_solver[t + 1].value = np.copy(self.Q[t+1])
            
            if self.problem_type == "oco":
                self.L_oco_accumulator.append(self.L[t])
            else:
                self.L_accumulator.append(self.L[t])

            self.f_in_accumulator.append(self.f_in_values[t])
            self.f_out_accumulator.append(self.f_out[t])
            self.Q_accumulator.append(self.Q[t+1])

            if self.problem_type == "eq":
                self.w_accumulator.append(self.w[t])
                self.prev_w[t+1] = self.w[t].value
                self.Q_solver_accumulator.append(self.Q_solver[t])
            else:
                self.w_accumulator.append(self.get_x_slices(var_name = "w", window_index = 1))
                self.prev_w[t+1] = self.get_x_slices(var_name = "w", window_index = 1).value
                self.Q_solver_accumulator.append(self.get_x_slices(var_name = "Q_solver", window_index = 1).value)

            self.delta_Q_accumulator.append(self.delta_Q[t])

        self.store_results()
        
    
    def get_x_slices(self, var_name, window_index, x_vec=None):
        """Returns a slice of x corresponding to the given variable name for the specified window."""
        # Define the order of variables
        var_order = ['f_in', 'w', 'L_solver', 'f_out_solver', 'Q_solver', 'delta_Q_solver']
        var_index = var_order.index(var_name)
        start_idx = var_index * M * P + window_index * NUM_COLS_PER_W
        end_idx = start_idx + M * P

        vector_to_slice = x_vec.transpose() if x_vec is not None else self.x
        return cp.vstack(vector_to_slice[start_idx:end_idx])
    

def save_to_file(filename, data, t):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    mode = 'w' if t == 0 else 'a'
    with open(filename, mode) as f:  # 'a' for append mode
        f.write(f"\n t = {t}\n")  # Write the label
        np.savetxt(f, data, fmt='%.1f')