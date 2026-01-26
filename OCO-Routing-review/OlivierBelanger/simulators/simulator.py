import numpy as np
import cvxpy as cp
from ..constants import *
class Simulator:
    def __init__(self):
        self.delta = None
        self.f_out = None
        self.Q_in = None
        self.Q_out = None
        self.delta_Q = None
        self.Q_next = None
        self.L = None
        self.epsilon = 1e-10

    def run_one_time_step(self, optimal_w, optimal_f_in, Q_curr, mpc_flows, realized_flows, t, problem_type):

        self.initialize_matrices()

        f_in_values, optimal_w = self.adjust_incoming_flow(optimal_f_in, optimal_w, mpc_flows, realized_flows, t, problem_type)

        for m in range(M):
            total_Q_curr = np.sum(Q_curr[:, m])
            temp_Q_curr = np.copy(Q_curr)
            available_space = max(Q_BAR - total_Q_curr, 0)

            for p in range(P):
                self.delta[p, m] = optimal_w[p, m] / DELTA_S - f_in_values[p][m]
                if self.delta[p, m] > 0:  # Underserved
                    self.handle_underserved(p, m, Q_curr)
                elif self.delta[p, m] < 0:   # Overserved
                    self.handle_overserved(p, m, temp_Q_curr, available_space)
                else: # Balanced
                    self.handle_balanced(p,m)
                self.update_queue_and_flow(p, m, Q_curr, f_in_values, optimal_w, t)
                
            self.ensure_feasibility(m)
            self.check_flow_conservation(f_in_values, p, m, t)
        self.check_output_flow(t, optimal_w)

        return self.f_out, self.Q_next, self.L, self.delta_Q, f_in_values
    

    def initialize_matrices(self):
        shape = (P, M)
        self.delta = np.zeros(shape)
        self.f_out = np.zeros(shape)
        self.Q_in = np.zeros(shape)
        self.Q_out = np.zeros(shape)
        self.delta_Q = np.zeros(shape)
        self.Q_next = np.zeros(shape)
        self.L = np.zeros(shape)
    
    def adjust_incoming_flow(self, optimal_f_in, optimal_w, mpc_flows, realized_flows, t, problem_type):
        if optimal_f_in is None:
            raise ValueError(f"t = {t}, optimal_f_in is None")

        safe_denominator = np.maximum(np.abs(mpc_flows[t]), self.epsilon)
        ratio = (realized_flows[t] / safe_denominator).reshape(P, 1)

        if problem_type in ["oco", "mat"]:
            optimal_f_in = np.array(optimal_f_in.value if isinstance(optimal_f_in, cp.Variable) else optimal_f_in).reshape(P, M)
            optimal_w = np.array(optimal_w.value).reshape(P, M)
        f_in_values = optimal_f_in * ratio

        return f_in_values, optimal_w
    
    def handle_underserved(self, p, m, Q_curr):
        self.Q_in[p, m]  = 0
        self.Q_out[p, m] = min(Q_curr[p, m], self.delta[p, m])
        self.L[p, m]     = 0

    def handle_overserved(self, p, m, temp_Q_curr, available_space):
        
        removed_packets = np.zeros((P, M))
        Q_in_from_delta = min(available_space, -self.delta[p, m])
        self.Q_in[p, m] += Q_in_from_delta
        extra = -self.delta[p, m] - Q_in_from_delta
        while extra > 0:
            removed = False
            for lower_p in reversed(range(p+1, P)):
                if temp_Q_curr[lower_p, m] > 0:
                    removed_packets = min(temp_Q_curr[lower_p, m], extra)
                    temp_Q_curr[lower_p, m] -= removed_packets
                    self.Q_out[lower_p, m] += removed_packets
                    self.L[lower_p, m] += removed_packets
                    self.Q_in[p, m] += removed_packets
                    extra -= removed_packets
                    removed = True
                    if extra <= 0:
                        break
            if not removed or extra <= 0:
                break
        self.L[p, m] += extra

    def handle_balanced(self, p, m):
        self.Q_in[p, m]  = 0
        self.Q_out[p, m] = 0
        self.L[p, m]     = 0

    def update_queue_and_flow(self, p, m, Q_curr, f_in_values, optimal_w, t):
        self.delta_Q[p, m] = self.Q_in[p, m] - self.Q_out[p, m]    
        self.f_out[p, m] = min(optimal_w[p, m] / DELTA_S, f_in_values[p][m] + self.Q_out[p, m])

        if t >= T - 2:
            self.Q_next[p, m] = Q_0
            self.L[p, m] += Q_curr[p, m] + self.delta_Q[p, m]
            self.delta_Q[p, m] = 0
        else:
            self.Q_next[p, m] = Q_curr[p, m] + self.delta_Q[p, m]


    def ensure_feasibility(self, m):
        sum_on_P = sum(self.Q_next[:P, m])
        if sum_on_P > Q_BAR:
            excess = sum_on_P - Q_BAR
            for p in reversed(range(P)):
                if excess <= 0:
                    break
                removal = min(self.Q_next[p, m], excess)
                self.Q_next[p, m] -= removal
                self.delta_Q[p, m] -= removal
                excess -= removal
                self.L[p, m] += removal
                if p == 0 and excess > 0:
                    self.L[p, m] += excess
                    excess = 0

    def check_flow_conservation(self, f_in_values, p, m, t):
        if (f_in_values[p,m] - self.delta_Q[p,m] - self.L[p,m] - self.f_out[p,m] > 1e-2):
            print("\nerror: flow conservation not satisfied")
            print("t = ", t, "p = ", p, "m = ", m)
            print("f_in_values[p,m] = ", f_in_values[p,m])
            print("f_out[p,m] = ", self.f_out[p,m])
            print("L[p,m] = ", self.L[p,m])
            print("Q_curr = ", self.Q_next[p,m] - self.delta_Q[p,m])  # Q_curr for next iteration
            print("Q_in = ", self.Q_in[p,m])
            print("Q_out = ", self.Q_out[p,m])
            print("delta_Q[p,m] = ", self.delta_Q[p,m])

    def check_output_flow(self, t, optimal_w):
        if self.f_out.any() <= 0:
            print("t = ", t)
            print("optimal_w = ", optimal_w)