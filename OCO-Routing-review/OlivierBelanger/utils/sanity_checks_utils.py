import numpy as np
import cvxpy as cp
from oop_opt.constants import *
# from oop_opt.mat_gen.matrices_gen import *
from oop_opt.problems.mpc_problem import *


def initialize_offline_accumulators():
    f_in_accumulator_offline = np.zeros((T, P))
    f_out_accumulator_offline = {t: np.zeros((P, M)) for t in range(T)}
    L_accumulator_offline = {t: np.zeros((P, M)) for t in range(T)}
    Q_accumulator_offline = {t: np.zeros((P, M)) for t in range(T)}
    delta_Q_accumulator_offline = {t: np.zeros((P, M)) for t in range(T)}
    return (f_in_accumulator_offline, f_out_accumulator_offline, L_accumulator_offline, Q_accumulator_offline, delta_Q_accumulator_offline)

def initialize_mpc_accumulators(problem_type):
    # f_in_accumulator_mpc = np.zeros((T, P))
    f_in_accumulator_mpc = {t: np.zeros((P, M)) for t in range(T)}
    f_out_accumulator_mpc = {t: np.zeros((P, M)) for t in range(T)}
    if problem_type == "mat" or problem_type == "oco":
        w_accumulator_mpc = [0.0]*T
        Q_solver_accumulator_mpc = [0.0]*T
    else:
        w_accumulator_mpc = {t: np.zeros((P, M)) for t in range(T)}
        Q_solver_accumulator_mpc = {t: np.zeros((P, M)) for t in range(T)}
    L_accumulator_mpc = {t: np.zeros((P, M)) for t in range(T)}
    Q_accumulator_mpc = {t: np.zeros((P, M)) for t in range(T)}
    delta_Q_accumulator_mpc = {t: np.zeros((P, M)) for t in range(T)}
    return (f_in_accumulator_mpc, f_out_accumulator_mpc, w_accumulator_mpc, L_accumulator_mpc, Q_accumulator_mpc, delta_Q_accumulator_mpc, Q_solver_accumulator_mpc)


def accumulate_offline_values(offline_problem, flows_offline):
    f_in_accumulator_offline, f_out_accumulator_offline, L_accumulator_offline, Q_accumulator_offline, delta_Q_accumulator_offline = initialize_offline_accumulators()
    # Accumulate f_out_values
    f_in_values_offline = flows_offline
    f_out_values_offline = offline_problem.f_out
    L_values_offline = offline_problem.L
    Q_values_offline = offline_problem.Q
    delta_Q_values_offline = offline_problem.delta_Q
    for t in range(T):
        f_in_accumulator_offline[t]  = cp.sum([f_in_accumulator_offline[t], f_in_values_offline[t]])
        f_out_accumulator_offline[t] = cp.sum([f_out_accumulator_offline[t], f_out_values_offline[t]])
        L_accumulator_offline[t]     = cp.sum([L_accumulator_offline[t], L_values_offline[t]])
        Q_accumulator_offline[t]     = cp.sum([Q_accumulator_offline[t], Q_values_offline[t]])
        delta_Q_accumulator_offline[t]  = cp.sum([delta_Q_accumulator_offline[t], delta_Q_values_offline[t]])
    
    return f_in_accumulator_offline, f_out_accumulator_offline, L_accumulator_offline, Q_accumulator_offline, delta_Q_accumulator_offline

def accumulate_mpc_values(mpc_problem, flows_mpc, problem_type, x = None):
    f_in_accumulator_mpc, f_out_accumulator_mpc, w_accumulator_mpc, L_accumulator_mpc, Q_accumulator_mpc, delta_Q_accumulator_mpc, Q_solver_accumulator_mpc = initialize_mpc_accumulators(problem_type)
    # Accumulate f_out_values
    
    # f_in_values_mpc = flows_mpc
    # f_in_values_mpc = mpc_problem.f_in_values
    # f_out_values_mpc = mpc_problem.f_out
    # w_values_mpc = mpc_problem.w
    # L_values_mpc = mpc_problem.L
    # Q_values_mpc = mpc_problem.Q
    # delta_Q_values_mpc = mpc_problem.delta_Q
    # Q_solver_values_mpc = mpc_problem.Q_solver

    f_in_values_mpc = mpc_problem.f_in_accumulator
    f_out_values_mpc = mpc_problem.f_out_accumulator
    L_values_mpc = mpc_problem.L_accumulator if problem_type == "eq" else mpc_problem.L_oco_accumulator
    Q_values_mpc = mpc_problem.Q_accumulator
    delta_Q_values_mpc = mpc_problem.delta_Q_accumulator
    Q_solver_values_mpc = mpc_problem.Q_solver_accumulator
    # print("sanity check = ", Q_solver_values_mpc)
    w_values_mpc = mpc_problem.w_accumulator
    # Q_solver_values_mpc = mpc_problem.Q_solver_accumulator
    
    for t in range(T):
        f_in_accumulator_mpc[t] += f_in_values_mpc[t]
        f_out_accumulator_mpc[t] += f_out_values_mpc[t]
        L_accumulator_mpc[t] += L_values_mpc[t]
        delta_Q_accumulator_mpc[t] += delta_Q_values_mpc[t]
        if problem_type == "eq":
            w_accumulator_mpc[t] += w_values_mpc[t].value
            Q_solver_accumulator_mpc[t] += Q_solver_values_mpc[t].value
            # print("sanity check = ", Q_solver_values_mpc[t].value)
        else:
            w_accumulator_mpc[t] += x.get_x_slices(var_name = "w", window_index = 1).value
            Q_solver_accumulator_mpc[t] += Q_solver_values_mpc[t]
            # print("sanity check = ", Q_solver_accumulator_mpc[t])
    for t in range(T-1):
        Q_accumulator_mpc[t+1] += Q_values_mpc[t]

    
    
    return f_in_accumulator_mpc, f_out_accumulator_mpc, w_accumulator_mpc, L_accumulator_mpc, Q_accumulator_mpc, delta_Q_accumulator_mpc, Q_solver_accumulator_mpc

def generate_offline_comparison_values(f_in_accumulator_offline, f_out_accumulator_offline, L_accumulator_offline, Q_accumulator_offline, delta_Q_accumulator_offline):
    # f_in, for comparison
    offline_avg_flows = np.around(f_in_accumulator_offline, 3)
    offline_avg_flows_sum_on_t = np.around(np.sum(offline_avg_flows, axis=0), 3)
    offline_avg_flows_total = round(np.sum(offline_avg_flows), 3)

    # f_in, for comparison, summed on m
    # offline_avg_flows = [cp.sum(f_in_accumulator_offline[t], axis=1) for t in range(T)]
    # offline_avg_flows_values = [expression.value for expression in offline_avg_flows]
    # offline_avg_flows_values_lists = [[round(num, 1) for num in arr.tolist()] for arr in offline_avg_flows_values]
    # offline_avg_flows_sum_on_t = np.around(np.sum([np.sum(f_in_accumulator_offline[t], axis=1) for t in range(T)], axis=0), 1)
    # offline_avg_flows_total = round(np.sum(offline_avg_flows_sum_on_t), 1)

    # L, for comparison
    offline_avg_L = [cp.sum(L_accumulator_offline[t], axis=1) for t in range(T)]
    offline_avg_L_values = [expression.value for expression in offline_avg_L]
    offline_avg_L_values_lists = [[round(num, 3) for num in arr.tolist()] for arr in offline_avg_L_values]
    offline_avg_L_sum_on_t = np.around(cp.sum([cp.sum(L_accumulator_offline[t], axis=1) for t in range(T)], axis=0).value, 3)
    offline_avg_L_total = round(np.sum(offline_avg_L_sum_on_t), 3)

    # f_out, for comparison
    offline_avg_f_out = [cp.sum(f_out_accumulator_offline[t], axis=1) for t in range(T)]
    offline_avg_f_out_values = [expression.value for expression in offline_avg_f_out]
    offline_avg_f_out_values_lists = [[round(num, 3) for num in arr.tolist()] for arr in offline_avg_f_out_values]
    offline_avg_f_out_sum_on_t = np.around(cp.sum([cp.sum(f_out_accumulator_offline[t], axis=1) for t in range(T)], axis=0).value, 3)
    offline_avg_f_out_total = round(np.sum(offline_avg_f_out_sum_on_t), 3)

    # Q, for comparison
    offline_avg_Q = [cp.sum(Q_accumulator_offline[t], axis=1) for t in range(T)]
    offline_avg_Q_values = [expression.value for expression in offline_avg_Q]
    offline_avg_Q_values_lists = [[round(num, 3) for num in arr.tolist()] for arr in offline_avg_Q_values]
    offline_last_Q_key = list(Q_accumulator_offline.keys())[-1]
    offline_avg_Q_total_end = round(cp.sum(Q_accumulator_offline[offline_last_Q_key]).value, 3)

    # delta_Q, for comparison
    offline_avg_delta_Q = [cp.sum(delta_Q_accumulator_offline[t], axis=1) for t in range(T)]
    offline_avg_delta_Q_values = [expression.value for expression in offline_avg_delta_Q]
    offline_avg_delta_Q_values_lists = [[round(num, 3) for num in arr.tolist()] for arr in offline_avg_delta_Q_values]
    offline_last_delta_Q_key = list(delta_Q_accumulator_offline.keys())[-1]
    offline_avg_delta_Q_total_end = round(cp.sum(delta_Q_accumulator_offline[offline_last_delta_Q_key]).value, 3)

    

    return offline_avg_flows, offline_avg_flows_sum_on_t, offline_avg_flows_total, offline_avg_L_values_lists, offline_avg_L_sum_on_t, offline_avg_L_total, offline_avg_f_out_values_lists, offline_avg_f_out_sum_on_t, offline_avg_f_out_total, offline_avg_Q_values_lists, offline_avg_Q_total_end, offline_avg_delta_Q_values_lists, offline_avg_delta_Q_total_end


def generate_mpc_comparison_values(f_in_accumulator_mpc, f_out_accumulator_mpc, w_accumulator_mpc, L_accumulator_mpc, Q_accumulator_mpc, delta_Q_accumulator_mpc, Q_solver_accumulator_mpc, problem_type):
    # f_in, for comparison
    # mpc_avg_flows = f_in_accumulator_mpc / NUM_SIM
    # mpc_avg_flows_sum_on_t = np.around(np.sum(mpc_avg_flows, axis=0), 1)
    # mpc_avg_flows_total = round(np.sum(mpc_avg_flows), 1)
    
    # f_in, for comparison, summed on m
    mpc_avg_flows = [cp.sum(f_in_accumulator_mpc[t], axis=1) for t in range(T)]
    mpc_avg_flows_values = [expression.value for expression in mpc_avg_flows]
    mpc_avg_flows_values_lists = [[round(num, 3) for num in arr.tolist()] for arr in mpc_avg_flows_values]
    mpc_avg_flows_sum_on_t = np.around(np.sum([np.sum(f_in_accumulator_mpc[t], axis=1) for t in range(T)], axis=0), 3)
    mpc_avg_flows_total = round(np.sum(mpc_avg_flows_sum_on_t), 3)

    # L, for comparison, summed on m
    mpc_avg_L = [cp.sum(L_accumulator_mpc[t], axis=1) for t in range(T)]
    mpc_avg_L_values = [expression.value for expression in mpc_avg_L]
    mpc_avg_L_values_lists = [[round(num, 3) for num in arr.tolist()] for arr in mpc_avg_L_values]
    mpc_avg_L_sum_on_t = np.around(np.sum([np.sum(L_accumulator_mpc[t], axis=1) for t in range(T)], axis=0), 3)
    mpc_avg_L_total = round(np.sum(mpc_avg_L_sum_on_t), 3)

    # f_out, for comparison, summed on m
    mpc_avg_f_out = [cp.sum(f_out_accumulator_mpc[t], axis=1) for t in range(T)]
    mpc_avg_f_out_values = [expression.value for expression in mpc_avg_f_out]
    mpc_avg_f_out_values_lists = [[round(num, 3) for num in arr.tolist()] for arr in mpc_avg_f_out_values]
    mpc_avg_f_out_sum_on_t = np.around(np.sum([np.sum(f_out_accumulator_mpc[t], axis=1) for t in range(T)], axis=0), 3)
    mpc_avg_f_out_total = round(np.sum(mpc_avg_f_out_sum_on_t), 3)

    # w, for comparison, summed on m
    mpc_avg_w = [w_accumulator_mpc[t] for t in range(T)]
    mpc_avg_w_values = [expression for expression in mpc_avg_w]
    mpc_avg_w_values_lists = [[[round(num, 3) for num in arr] for arr in expression.tolist()] for expression in mpc_avg_w_values]
    mpc_avg_w_sum_on_t = np.around([np.sum(w_accumulator_mpc[t]) for t in range(T)], 3) 
    mpc_avg_w_total = round(np.sum(mpc_avg_w_sum_on_t), 3)
    
    # Q, for comparison
    mpc_avg_Q = [cp.sum(Q_accumulator_mpc[t], axis=1) for t in range(T)]
    mpc_avg_Q_values = [expression.value for expression in mpc_avg_Q]
    mpc_avg_Q_values_lists = [[round(num, 3) for num in arr.tolist()] for arr in mpc_avg_Q_values]
    mpc_last_Q_key = list(Q_accumulator_mpc.keys())[-1]
    mpc_avg_Q_total_end = round(cp.sum(Q_accumulator_mpc[mpc_last_Q_key]).value, 3)

    # delta_Q, for comparison
    mpc_avg_delta_Q = [cp.sum(delta_Q_accumulator_mpc[t], axis=1) for t in range(T)]
    mpc_avg_delta_Q_values = [expression.value for expression in mpc_avg_delta_Q]
    mpc_avg_delta_Q_values_lists = [[round(num, 3) for num in arr.tolist()] for arr in mpc_avg_delta_Q_values]
    mpc_last_delta_Q_key = list(delta_Q_accumulator_mpc.keys())[-1]
    mpc_avg_delta_Q_total_end = round(cp.sum(delta_Q_accumulator_mpc[mpc_last_delta_Q_key]).value, 3)

    # Q, for comparison
    mpc_avg_Q_solver = [cp.sum(Q_solver_accumulator_mpc[t], axis=1) for t in range(T)]
    mpc_avg_Q_solver_values = [expression.value for expression in mpc_avg_Q_solver]
    mpc_avg_Q_solver_values_lists = [[round(num, 3) for num in arr.tolist()] for arr in mpc_avg_Q_solver_values]
    if problem_type == "eq":
        mpc_last_Q_solver_key = list(Q_solver_accumulator_mpc.keys())[-1]
        mpc_avg_Q_solver_total_end = round(cp.sum(Q_solver_accumulator_mpc[mpc_last_Q_solver_key]).value, 3)
    else:
        # print("sanity check - Q_solver_accumulator_mpc", Q_solver_accumulator_mpc)
        mpc_last_Q_solver_index = -1
        mpc_avg_Q_solver_total_end = round(cp.sum(Q_solver_accumulator_mpc[mpc_last_Q_solver_index]).value, 3)

    return mpc_avg_flows_values_lists, mpc_avg_flows_sum_on_t, mpc_avg_flows_total, mpc_avg_f_out_values_lists, mpc_avg_f_out_sum_on_t, mpc_avg_f_out_total, mpc_avg_w_values_lists, mpc_avg_w_sum_on_t, mpc_avg_w_total, mpc_avg_L_values_lists, mpc_avg_L_sum_on_t, mpc_avg_L_total, mpc_avg_Q_values_lists, mpc_avg_Q_total_end, mpc_avg_delta_Q_values_lists, mpc_avg_delta_Q_total_end, mpc_avg_Q_solver_values_lists, mpc_avg_Q_solver_total_end
