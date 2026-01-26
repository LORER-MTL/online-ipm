import numpy as np
import pickle as pk
import time

from oop_opt.problems.offline_problem import OfflineProblem
from oop_opt.problems.mpc_problem import MPCProblem
from oop_opt.utils.plot_utils import *
from oop_opt.utils.generate_flow import *
from oop_opt.utils.sanity_checks_utils import *
from oop_opt.simulators.simulator import *
from oop_opt.constants import *
from oop_opt.ocotools.Problem import *


def load_data_from_file(filename):
    with open(filename, 'rb') as file:
        return pk.load(file)
    

def initialize_arrays():
    # Initialize arrays to store results
    offline_L_values = {t: np.zeros((P, M)) for t in range(T)}
    prop_L_values = {t: np.zeros((P, M)) for t in range(T)}
    mpc_L_values = {t: np.zeros((P, M)) for t in range(T)}
    ocmpc_L_values = {t: np.zeros((P, M)) for t in range(T)}
    static_batch_avg_L_values = {t: np.zeros((P, M)) for t in range(T)}
    flows_accumulator = np.zeros((T+W, P))
    flows_mpc_accumulator = np.zeros((T+W, P))
    return (offline_L_values, prop_L_values, mpc_L_values, ocmpc_L_values, static_batch_avg_L_values, flows_accumulator, flows_mpc_accumulator)

def run_w_comp():
    mpc_results = {}
    mpc_data = {}
    for i in range(NUM_SIM):
        realized_flow, flow = generate_flows(flow_type='mmpp')
        for current_W in W_VALUES:
            print(f"Round {i+1}, MPC, W = {current_W}")
            flows_mpc_realized, flows_mpc = realized_flow, flow

            mpc_problem = MPCProblem("MPCProblem", start_t=0, end_t=T, flows=flows_mpc, realized_flows=flows_mpc_realized, W=current_W, strategy="mpc")
            mpc_problem.manage()

            if current_W not in mpc_results:
                mpc_results[current_W] = []
            mpc_results[current_W].append(np.array(mpc_problem.L_accumulator))

            f_in_accumulator_mpc, f_out_accumulator_mpc, w_accumulator_mpc, L_accumulator_mpc, Q_accumulator_mpc, delta_Q_accumulator_mpc, Q_solver_accumulator_mpc = accumulate_mpc_values(mpc_problem, flows_mpc)
            if current_W not in mpc_data:
                mpc_data[current_W] = []
            mpc_data[current_W].append((f_in_accumulator_mpc, f_out_accumulator_mpc, w_accumulator_mpc, L_accumulator_mpc, Q_accumulator_mpc, delta_Q_accumulator_mpc, Q_solver_accumulator_mpc))
    with open('W_comparison.txt', 'w') as f:
        f.write(f"Constants: T = {T}, P = {P}, M = {M}\n\n")
        for current_W in W_VALUES:
            mpc_results[current_W] = np.mean(mpc_results[current_W], axis=0)

            mpc_avg_flows_values_lists, mpc_avg_flows_sum_on_t, mpc_avg_flows_total, mpc_avg_f_out_values_lists, mpc_avg_f_out_sum_on_t, mpc_avg_f_out_total, mpc_avg_w_values_lists, mpc_avg_w_sum_on_t, mpc_avg_w_total, mpc_avg_L_values_lists, mpc_avg_L_sum_on_t, mpc_avg_L_total, mpc_avg_Q_values_lists, mpc_avg_Q_total_end, mpc_avg_delta_Q_values_lists, mpc_avg_delta_Q_total_end, mpc_avg_Q_solver_values_lists, mpc_avg_Q_solver_total_end = generate_mpc_comparison_values(*mpc_data[current_W][0])
            
            f.write(f"########## MPC, W = {current_W} ##########\n")  # Print current W value
            data_mpc = list(zip(mpc_avg_flows_values_lists, mpc_avg_f_out_values_lists, mpc_avg_w_values_lists, mpc_avg_L_values_lists, mpc_avg_Q_values_lists, mpc_avg_Q_solver_values_lists, mpc_avg_delta_Q_values_lists, ))
            f.write(f"{'t':<5}{'f_in':<25}{'f_out':<25}{'w':<50}{'L':<25}{'Q':<25}{'Q_solver':<25}{'delta_Q':<25}\n")
            for t, row in enumerate(data_mpc):
                f.write(f"{t:<5}{str(row[0]):<25}{str(row[1]):<25}{str(row[2]):<50}{str(row[3]):<25}{str(row[4]):<25}{str(row[5]):<25}{str(row[6]):<25}\n")

            f.write(f"{'Sum':<5}{str(mpc_avg_flows_sum_on_t):<25}{str(mpc_avg_f_out_sum_on_t):50}{str(mpc_avg_L_sum_on_t):<25}\n")
            f.write(f"{'Total':<5}{str(mpc_avg_flows_total):<25}{str(mpc_avg_f_out_total):50}{str(mpc_avg_L_total):<25}\n")
            f.write(f"{'Total @ t=T':<80}{str(mpc_avg_Q_total_end):<25}{str(mpc_avg_delta_Q_total_end):<25}\n")

    plot_mpc_packet_loss_across_W_values(mpc_results, W_VALUES)


def run_scenario_comp(reuse_data=False, data_file='test.dat'):
    
    if reuse_data:
        full_data_file = 'oop_opt/results/' + data_file
        loaded_data = load_data_from_file(full_data_file)
        offline_L_array = loaded_data['offline_L_array']
        prop_L_array = loaded_data['prop_L_array']
        mpc_L_array = loaded_data['mpc_L_array']
        ocmpc_L_array = loaded_data['ocmpc_L_array']
    else:
        offline_L_values, prop_L_values, mpc_L_values, ocmpc_L_values, static_batch_avg_L_values, flows_accumulator, flows_mpc_accumulator = initialize_arrays()
        offline_L_values_list = []
        prop_L_values_list = []
        mpc_L_values_list = []
        ocmpc_L_values_list = []

        global_start_time = time.time()
        for i in range(NUM_SIM):
            offline_L_values, prop_L_values, mpc_L_values, ocmpc_L_values = initialize_arrays()[:4]
            print("Round ", i+1)
            
            realized_flows, expected_flows = generate_flows(flow_type='mmpp')
            run_problems(realized_flows, expected_flows, offline_L_values, prop_L_values, mpc_L_values, ocmpc_L_values)
            flows_accumulator += realized_flows
            flows_mpc_accumulator += expected_flows

            offline_L_values_list.append(offline_L_values)
            prop_L_values_list.append(prop_L_values)
            mpc_L_values_list.append(mpc_L_values)
            ocmpc_L_values_list.append(ocmpc_L_values)
  
        print("Total time elapsed: ", time.time() - global_start_time)

        # ################################ Plotting ################################# 
        offline_L_array = dict_list_to_array(offline_L_values_list)
        prop_L_array = dict_list_to_array(prop_L_values_list)
        mpc_L_array = dict_list_to_array(mpc_L_values_list)
        ocmpc_L_array = dict_list_to_array(ocmpc_L_values_list)

    plot_total_cost_comp_uncertainty(offline_L_array, prop_L_array, mpc_L_array, ocmpc_L_array, uncertainty_type = "percentile")
    # plot_flows_vs_time(flows_accumulator[:T, :]/NUM_SIM, flows_mpc_accumulator[:T, :]/NUM_SIM) #Avg on N Monte-Carlo runs
    # plot_flows_vs_time(flows_mpc_realized[:T, :], flows_mpc[:T, :]) #Single run

    # ################################  OLD CODE ################################# 
    # offline_L_avg = {t: v / NUM_SIM for t, v in offline_L_values.items()}
    # fixed_L_avg = {t: v / NUM_SIM for t, v in fixed_L_values.items()}
    # prop_L_avg = {t: v / NUM_SIM for t, v in prop_L_values.items()}
    # mpc_L_avg = {t: v / NUM_SIM for t, v in mpc_L_values.items()}
    # static_batch_avg_L_avg = {t: v / NUM_SIM for t, v in static_batch_avg_L_values.items()}
    
    # plot_total_cost_comp(offline_L_avg, prop_L_avg, mpc_L_avg, fixed_L_avg)
    # plot_total_cost_comp(offline_L_avg, prop_L_avg, mpc_L_avg, fixed_L_avg, static_batch_avg_L_avg)
    # plot_packet_loss_across_time(offline_L_avg, fixed_L_avg, prop_L_avg, mpc_L_avg, static_batch_avg_L_avg)

    # INCOMING FLOWS PLOTTING
    #CAREFUL: mpc_avg_flows_values_lists IS THE REALIZED FLOW (ADJUSTED AFTER RATIO), NOT SIMPLY THE AVERAGE
    # print(flows_accumulator.shape, flows_mpc_accumulator[:T].shape)
    # # OUTGOING FLOWS PLOTTING
    # averaged_f_out = {t: f_out_accumulator_mpc[t] / NUM_SIM for t in range(T)}
    # f_out_array = convert_and_sum_across_modules(averaged_f_out, T, P)
    # plot_summed_outgoing_flows_vs_time(f_out_array, P)


def run_problems(realized_flows, expected_flows, offline_L_values, prop_L_values, mpc_L_values, ocmpc_L_values):
    """Run the main problem simulations for offline, proportional, MPC, and OCMPC strategies."""
    print("OCMPC, W =", W)
    start_time = time.time()
    ocmpc_problem_mat = MPCProblem("MPCProblem", start_t=0, end_t=T, flows=expected_flows, realized_flows=realized_flows, W=W, strategy="mpc", problem_type = "oco")
    ocmpc_problem_mat.manage()
    print("Time elapsed: ", time.time() - start_time)

    print("Batch w/ hindsight")
    start_time = time.time()
    offline_problem = OfflineProblem("OfflineProblem", start_t=0, end_t=T, flows=realized_flows, strategy="dynamic")
    offline_problem.solve()
    print("Time elapsed: ", time.time() - start_time)

    print("Proportional")
    start_time = time.time()
    prop_problem = OfflineProblem("ProportionalProblem", start_t=0, end_t=T, flows=realized_flows, strategy="proportional")
    prop_problem.solve()
    print("Time elapsed: ", time.time() - start_time)

    print("MPC, W =", W)
    start_time = time.time()
    mpc_problem = MPCProblem("MPCProblem", start_t=0, end_t=T, flows=expected_flows, realized_flows=realized_flows, W=W, strategy="mpc")
    mpc_problem.manage()
    print("Time elapsed: ", time.time() - start_time)

    accumulate_L_values(T, offline_L_values, offline_problem.L_values)
    accumulate_L_values(T, prop_L_values, prop_problem.L_values)
    accumulate_L_values(T, mpc_L_values, mpc_problem.L_accumulator)
    accumulate_L_values(T, ocmpc_L_values, ocmpc_problem_mat.L_oco_accumulator)

    # Accumulate values for data gathering
    f_in_accumulator_offline, f_out_accumulator_offline, L_accumulator_offline, Q_accumulator_offline, delta_Q_accumulator_offline = accumulate_offline_values(offline_problem, realized_flows)
    f_in_accumulator_mpc, f_out_accumulator_mpc, w_accumulator_mpc, L_accumulator_mpc, Q_accumulator_mpc, delta_Q_accumulator_mpc, Q_solver_accumulator_mpc = accumulate_mpc_values(mpc_problem, expected_flows, problem_type="eq", x=None)
    scenario_data = {
        'offline': generate_offline_comparison_values(f_in_accumulator_offline, f_out_accumulator_offline, L_accumulator_offline, Q_accumulator_offline, delta_Q_accumulator_offline),
        'mpc': generate_mpc_comparison_values(f_in_accumulator_mpc, f_out_accumulator_mpc, w_accumulator_mpc, L_accumulator_mpc, Q_accumulator_mpc, delta_Q_accumulator_mpc, Q_solver_accumulator_mpc, problem_type="eq")
    }
    write_scenario_comp_data(scenario_data)


def write_scenario_comp_data(scenario_data, filename='scenario_comparison.txt'):
    with open(filename, 'w') as f:
        f.write(f"Constants: T = {T}, W = {W}, P = {P}, M = {M}\n\n")

        for scenario_type in ['offline', 'mpc']:
            data = scenario_data[scenario_type]
            f.write(f"{'#' * 10} {scenario_type.upper()} {'#' * 10}\n\n")

            if scenario_type == 'offline':
                (avg_flows, avg_flows_sum_on_t, avg_flows_total, 
                 avg_L_values, avg_L_sum_on_t, avg_L_total, 
                 avg_f_out_values, avg_f_out_sum_on_t, avg_f_out_total, 
                 avg_Q_values, avg_Q_total_end, 
                 avg_delta_Q_values, avg_delta_Q_total_end) = data

                headers = ['t', 'f_in', 'f_out', 'L', 'Q', 'delta_Q']
                f.write(f"{headers[0]:<5}" + "".join(f"{h:<25}" for h in headers[1:]) + "\n")
                for t in range(len(avg_flows)):
                    row = [avg_flows[t], avg_f_out_values[t], avg_L_values[t], avg_Q_values[t], avg_delta_Q_values[t]]
                    f.write(f"{t:<5}" + "".join(f"{str(val):<25}" for val in row) + "\n")
                f.write(f"{'Sum':<5}{str(avg_flows_sum_on_t):<25}{str(avg_f_out_sum_on_t):<25}{str(avg_L_sum_on_t):<25}\n")
                f.write(f"{'Total':<5}{str(avg_flows_total):<25}{str(avg_f_out_total):<25}{str(avg_L_total):<25}\n")
                f.write(f"{'Total @ t=T':<80}{str(avg_Q_total_end):<25}{str(avg_delta_Q_total_end):<25}\n\n")
            
            elif scenario_type == 'mpc':
                (avg_flows_values, avg_flows_sum_on_t, avg_flows_total, 
                 avg_f_out_values, avg_f_out_sum_on_t, avg_f_out_total, 
                 avg_w_values, avg_w_sum_on_t, avg_w_total, 
                 avg_L_values, avg_L_sum_on_t, avg_L_total, 
                 avg_Q_values, avg_Q_total_end, 
                 avg_delta_Q_values, avg_delta_Q_total_end, 
                 avg_Q_solver_values, avg_Q_solver_total_end) = data

                headers = ['t', 'f_in', 'f_out', 'w', 'L', 'Q', 'Q_solver', 'delta_Q']
                f.write(f"{headers[0]:<5}" + "".join(f"{h:<25}" for h in headers[1:]) + "\n")
                for t in range(len(avg_flows_values)):
                    row = [avg_flows_values[t], avg_f_out_values[t], avg_w_values[t], avg_L_values[t], 
                           avg_Q_values[t], avg_Q_solver_values[t], avg_delta_Q_values[t]]
                    f.write(f"{t:<5}" + "".join(f"{str(val):<25}" for val in row) + "\n")
                f.write(f"{'Sum':<5}{str(avg_flows_sum_on_t):<25}{str(avg_f_out_sum_on_t):<25}{str(avg_w_sum_on_t):<25}{str(avg_L_sum_on_t):<25}\n")
                f.write(f"{'Total':<5}{str(avg_flows_total):<25}{str(avg_f_out_total):<25}{str(avg_w_total):<25}{str(avg_L_total):<25}\n")
                f.write(f"{'Total @ t=T':<80}{str(avg_Q_total_end):<25}{str(avg_Q_solver_total_end):<25}{str(avg_delta_Q_total_end):<25}\n\n")


def accumulate_L_values(T, L_values, new_L_values):
    """Accumulate L values across all time steps."""
    for t in range(T):
        L_values[t] += new_L_values[t]