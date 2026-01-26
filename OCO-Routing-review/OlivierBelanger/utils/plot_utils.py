import datetime
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from ..constants import *


def configure_plot():
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"]= "serif"
    plt.rc('text.latex', preamble=r'\usepackage{bm}')
    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.labelweight': 'bold'})

configure_plot()

def set_standard_plot_settings(xlabel, ylabel, set_xticks, ax=None):
    if ax is None:
        ax = plt.gca() 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax.minorticks_on() 

    ax.autoscale(enable=True, axis='both', tight=True)
    ax.set_ylim(bottom=0)

    if set_xticks:
        xticks = np.arange(0, T+1, 1)
        xtick_labels = [f'{x:.0f}' for x in xticks]
        plt.xticks(xticks, xtick_labels)
    
def generate_timestamped_filename(prefix, extension='.pdf'):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"oop_opt/results/{prefix}_{timestamp}{extension}"

def save_and_show_plot(filename):
    plt.savefig(filename, dpi=800)
    plt.show()


def get_plot_styles(num_styles):
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))]
    colors = default_colors[:num_styles]
    styles = linestyles[:num_styles]

    return colors, styles

def compute_cumulative_sum(loss_values, is_uncertainty=False):

    if is_uncertainty:
        results = np.zeros((NUM_SIM, T))
        for sim in range(NUM_SIM):
            cumulative_sum = 0
            for t in range(T):
                current_sum = np.sum(loss_values[sim, t], axis=1)  # Sum along m dimension
                cumulative_sum += np.dot(current_sum, k_p)  # Update cumulative sum
                results[sim, t] = cumulative_sum
    else:
        results = np.zeros(T)
        cumulative_sum = 0
        for t in range(T):
            current_sum = np.sum(loss_values[t], axis=1)  # Sum along the m dimension
            cumulative_sum += np.dot(current_sum, k_p)  # Update cumulative sum
            results[t] = cumulative_sum

    return results




def print_and_save_comparison_matrix(results, methods, filename, range_size):
    reference_index = -1 if range_size == 1 else 0
    performance_matrix = np.full((range_size, range_size), np.nan)

    # Performance percentages
    for i in range(range_size):
        if i != reference_index:
            ratio = results[i] / results[reference_index]
            performance_percentage = (ratio - 1) * 100
            performance_matrix[i, reference_index] = performance_percentage

    # Width for formatting
    max_method_length = max(map(len, methods))
    width = max(max_method_length, 7)

    print(" ".ljust(width), end="  ")
    print("  ".join([method.ljust(width) for method in methods]))
    print("-" * (width + 2) * (range_size + 1))  

    with open(filename + '.txt', 'w') as txtfile:
        txtfile.write(" ".ljust(width) + "  ")
        txtfile.write("  ".join([method.ljust(width) for method in methods]) + "\n")
        for i in range(range_size):
            row_values = ["{:.2f}%".format(performance_matrix[i, j]).ljust(width) if not np.isnan(performance_matrix[i, j]) else " ".ljust(width) for j in range(range_size)]
            print(methods[i].ljust(width), end="  ")
            print("  ".join(row_values))
            txtfile.write(methods[i].ljust(width) + "  " + "  ".join(row_values) + "\n")
            if i < range_size - 1:
                print("-" * (width + 2) * (range_size + 1))
                txtfile.write("-" * (width + 2) * (range_size + 1) + "\n")

def convert_and_sum_across_modules(f_out_values, T, P):
    """Converts the f_out_values dictionary to a numpy array and sums across modules."""
    arr = np.zeros((T, P))
    for t in range(T):
        arr[t] = f_out_values[t].sum(axis=1)
    return arr


def plot_flows_vs_time(flows, flows_mpc):
    plt.figure(figsize=(10, 7))
    time_steps = np.arange(T)  
    colors, linestyles = get_plot_styles(P)

    flows_mpc = np.array(flows_mpc) 
    for p in range(P):
        sns.lineplot(x=time_steps, y=flows[:, p], color=colors[p], linestyle=linestyles[0], label=f'Observed, p = {p+1}', linewidth=2.5) 
        sns.lineplot(x=time_steps, y=flows_mpc[:, p], color=colors[p], linestyle=linestyles[1], label=f'Expected, p = {p+1}', linewidth=2.5)
    
    plt.legend(loc='upper left')
    set_standard_plot_settings('Time step $t$', 'Incoming flow', False)
    filename = generate_timestamped_filename('flows_vs_time_')
    save_and_show_plot(filename)


def plot_summed_outgoing_flows_vs_time(f_out_array, P):
    plt.figure(figsize=(10, 7))
    time_steps = np.arange(f_out_array.shape[0])
    colors, linestyles = get_plot_styles(P)

    for p in range(P):
        sns.lineplot(x=time_steps, y=f_out_array[:, p], color=colors[p], linestyle=linestyles[0], label=f'Priority {p+1}', linewidth=2.5)

    set_standard_plot_settings('Time step $t$', 'Outgoing flow', False)
    plt.legend(loc='lower right')
    filename = generate_timestamped_filename('summed_outgoing_flows_vs_time_')
    save_and_show_plot(filename)

def get_zoomed_inset_ranges(data_lists, x1, x2):
    y_min = min(min(data_list[x1:x2+1]) for data_list in data_lists)
    y_max = max(max(data_list[x1:x2+1]) for data_list in data_lists)
    return y_min, y_max


def plot_total_cost_comp(L_values, L_prop_values, L_mpc_values, L_ocmpc_values, additional_costs=None):
    # Cumulative sums
    packet_loss_cost_cumsum                     = compute_cumulative_sum(L_values, is_uncertainty=False)
    packet_loss_cost_prop_cumsum                = compute_cumulative_sum(L_prop_values, is_uncertainty=False)
    # packet_loss_cost_fixed_cumsum               = compute_cumulative_sum(L_fixed_values, is_uncertainty=False)
    packet_loss_cost_mpc_cumsum                 = compute_cumulative_sum(L_mpc_values, is_uncertainty=False)
    packet_loss_cost_ocmpc_cumsum             = compute_cumulative_sum(L_ocmpc_values, is_uncertainty=False)

    if additional_costs is not None:
        additional_costs_cumsum = compute_cumulative_sum(additional_costs, is_uncertainty=False)

    results = [
        packet_loss_cost_cumsum[-1],
        packet_loss_cost_prop_cumsum[-1], 
        packet_loss_cost_mpc_cumsum[-1], 
        packet_loss_cost_ocmpc_cumsum[-1]
    ]

    methods = ["Batch with hindsight", "Cost-based proportional", "MPC", "OCMPC"]

    if additional_costs is not None:
        results.append(additional_costs_cumsum[-1])
        methods.append("Additional Method")

    colors, linestyles = get_plot_styles(5 if additional_costs is None else 6) 
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    for idx, packet_loss_cost in enumerate([packet_loss_cost_cumsum, packet_loss_cost_prop_cumsum, packet_loss_cost_mpc_cumsum, packet_loss_cost_ocmpc_cumsum]):
        ax.plot(range(T), packet_loss_cost, linestyle=linestyles[idx], color=colors[idx], label=methods[idx], linewidth=2.5)

    if additional_costs is not None:
        ax.plot(range(T), additional_costs_cumsum, linestyle=linestyles[-1], color=colors[-1], label=methods[-1], linewidth=2.5)

    # Set y-axis limit
    max_ocmpc = max(packet_loss_cost_ocmpc_cumsum)
    print("max_ocmpc:", max_ocmpc)

    set_standard_plot_settings('Time step $t$', 'Cumulative Packet Loss Cost', False)
    ax.set_ylim(0, max_ocmpc * 1.05)  # 5% above the max value of the OCMPC curve
    plt.legend()

    # Zoomed box
    axins = inset_axes(ax, width='30%', height='30%', loc='center', bbox_to_anchor=(0.7, 0.0, 0.4, 0.4), bbox_transform=ax.transAxes)
    axins.patch.set_edgecolor('black') 
    axins.patch.set_linewidth(1)  

    # Plot same data in inset for zoomed view
    for idx, packet_loss_cost in enumerate([packet_loss_cost_cumsum, packet_loss_cost_mpc_cumsum, packet_loss_cost_ocmpc_cumsum]):
        color_idx = colors[idx] if idx == 0 else colors[idx+1]
        axins.plot(range(T), packet_loss_cost, linestyle=linestyles[idx], color=color_idx, linewidth=2.5)

    if additional_costs is not None:
        axins.plot(range(T), additional_costs_cumsum, linestyle=linestyles[-1], color=colors[-1], linewidth=2.5)

    # Zoom range for inset
    x1, x2 = T-2, T-1
    y1, y2 = get_zoomed_inset_ranges([packet_loss_cost_cumsum, packet_loss_cost_mpc_cumsum], x1, x2)
    if additional_costs is not None:
        y1_add, y2_add = get_zoomed_inset_ranges([additional_costs_cumsum], x1, x2)
        y1 = min(y1, y1_add)
        y2 = max(y2, y2_add)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    filename = generate_timestamped_filename(f'comparison_kp_{k_p}_M_{M}_P_{P}_Qbar_{Q_BAR}_W_{W}_T_{T}_dS_{DELTA_S}_dW_{DELTA_W_BAR}_NUMSIM_{NUM_SIM}')
    print_and_save_comparison_matrix(results, methods, filename, range_size = 4)
    save_and_show_plot(filename)


def plot_total_cost_comp_uncertainty(L_values, L_prop_values, L_mpc_values, L_ocmpc_values, uncertainty_type="none"):
    # Compute cumulative sums
    packet_loss_cost_cumsum = compute_cumulative_sum(L_values, is_uncertainty=True)
    packet_loss_cost_prop_cumsum = compute_cumulative_sum(L_prop_values, is_uncertainty=True)
    packet_loss_cost_mpc_cumsum = compute_cumulative_sum(L_mpc_values, is_uncertainty=True)
    packet_loss_cost_ocmpc_cumsum = compute_cumulative_sum(L_ocmpc_values, is_uncertainty=True)

    # Compute means and uncertainty bounds
    methods = ["Batch with hindsight", "Cost-based proportional", "MPC", "OCMPC"]
    cumsum_data = [packet_loss_cost_cumsum, packet_loss_cost_prop_cumsum, 
                   packet_loss_cost_mpc_cumsum, packet_loss_cost_ocmpc_cumsum]

    if uncertainty_type == "std":
        means_and_uncertainties = [compute_mean_and_std(data) for data in cumsum_data]
    elif uncertainty_type == "percentile":
        means_and_uncertainties = [compute_mean_and_percentiles(data) for data in cumsum_data]
    elif uncertainty_type == "none":
        means_and_uncertainties = [(np.mean(data, axis=0), None, None) for data in cumsum_data]
    else:
        raise ValueError("Invalid uncertainty_type. Choose 'std', 'percentile', or 'none'.")

    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    colors, linestyles = get_plot_styles(len(methods))

    for idx, (mean, lower, upper) in enumerate(means_and_uncertainties):
        ax.plot(range(T), mean, linestyle=linestyles[idx], color=colors[idx], label=methods[idx], linewidth=2.5)
        if uncertainty_type != "none":
            ax.fill_between(range(T), lower, upper, color=colors[idx], alpha=0.1)

    max_ocmpc_mean = np.max(means_and_uncertainties[-1][0])  # Mean of OCMPC

    set_standard_plot_settings('Time step $t$', 'Cumulative Packet Loss Cost', False)
    plt.legend()

    # zoomed inset
    ax.set_ylim(0, max_ocmpc_mean * 1.05)  # 5% above the max value of the OCMPC curve
    plt.legend()
    # Add minor ticks to the y-axis
    # ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))  # Adjust 0.5 to desired interval for minor ticks
    # ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))  # Format minor ticks

    axins = inset_axes(ax, width='30%', height='30%', loc='center', bbox_to_anchor=(0.7, 0.0, 0.4, 0.4), bbox_transform=ax.transAxes)
    axins.patch.set_edgecolor('black') 
    axins.patch.set_linewidth(1)  

    for idx, (mean, _, _) in enumerate([means_and_uncertainties[0], means_and_uncertainties[2], means_and_uncertainties[3]]):  # Batch, MPC, OCMPC
        color_idx = colors[idx] if idx == 0 else colors[idx+1]
        axins.plot(range(T), mean, linestyle=linestyles[idx], color=color_idx, linewidth=2.5)

    # Zoom range for inset
    x1, x2 = T-2, T-1
    y1, y2 = get_zoomed_inset_ranges([stats[0] for stats in [means_and_uncertainties[0], means_and_uncertainties[2]]], x1, x2)
    
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticks([])
    yticks = np.linspace(y1, y2, num=3)
    yticks_rounded = np.round(yticks).astype(int)
    axins.set_yticks(yticks_rounded)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # Mean values at the last time step for comparison
    results = [mean[-1] for mean, _, _ in means_and_uncertainties]

    
    ###SAVE DATA TO DAT FILE
    filename_dat = generate_timestamped_filename(f'sim_data', extension = '.dat')
    data_to_save = {
        'offline_L_array': L_values,
        'prop_L_array': L_prop_values,
        'mpc_L_array': L_mpc_values,
        'ocmpc_L_array': L_ocmpc_values
    }
    save_data_to_file(data_to_save, filename_dat)

    filename = generate_timestamped_filename(f'comparison_{uncertainty_type}_kp_{k_p}_M_{M}_P_{P}_Qbar_{Q_BAR}_W_{W}_T_{T}_dS_{DELTA_S}_dW_{DELTA_W_BAR}_NUMSIM_{NUM_SIM}')
    print_and_save_comparison_matrix(results, methods, filename, range_size=4)
    save_and_show_plot(filename)

def dict_list_to_array(dict_list):
    T = len(dict_list[0])  
    return np.array([[d[t] for t in range(T)] for d in dict_list])

def compute_mean_and_percentiles(data, percentile=95):
    mean = np.mean(data, axis=0)
    lower = np.percentile(data, (100 - percentile) / 2, axis=0)
    upper = np.percentile(data, 100 - (100 - percentile) / 2, axis=0)
    return mean, lower, upper


def compute_mean_and_std(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mean, mean - std, mean + std

def compute_cumulative_packet_loss(loss_values):
    cumsum_values = []
    for p in range(P):
        cumsum_values.append(np.cumsum([sum(loss_values[t][p]) for t in sorted(loss_values.keys())]))
    return np.array(cumsum_values)


def plot_packet_loss_across_time(L_values, L_fixed_values, L_prop_values, L_mpc_values, L_static_batch_mean_values):
    
    methods = ["Batch with hindsight", "Static batch with hindsight", "Proportional", "MPC", "MPC mat"]
    L_values_list = [L_values, L_fixed_values, L_prop_values, L_mpc_values, L_static_batch_mean_values]
    colors, linestyles = get_plot_styles(len(L_values_list))

    _, ax = plt.subplots(figsize=(10, 7))
    priority_legend_elements = []

    for scenario_index, L_values_scenario in enumerate(L_values_list):
        L_values_cumsum = compute_cumulative_packet_loss(L_values_scenario)

        color = colors[scenario_index]
        
        for priority in range(P):
            linestyle = linestyles[priority % len(linestyles) + 1]
            ax.plot(range(T), L_values_cumsum[priority], linestyle=linestyle, color=color, linewidth=2.5)
            if scenario_index == 0:
                priority_legend_elements.append(Line2D([0], [0], linestyle=linestyle, color='black', label=f'Priority {priority + 1}'))

        summed_L_values = np.sum(L_values_cumsum, axis=0)
        ax.plot(range(T), summed_L_values, linestyle='-', color=color, linewidth=2.5)

    scenario_legend_elements = [Line2D([0], [0], color=color, linestyle='-', label=method) for color, method in zip(colors, methods)]
    priority_legend_elements.append(Line2D([0], [0], color='black', linestyle='-', label='Cumulative'))

    legend1 = ax.legend(handles=scenario_legend_elements, loc="upper left", title="Scenarios")
    plt.gca().add_artist(legend1)
    ax.legend(handles=priority_legend_elements, loc="upper left", bbox_to_anchor=(0, 1 - legend1.get_window_extent().transformed(ax.transAxes.inverted()).height), title="Priorities")

    # Zoomed Box
    axins = inset_axes(ax, width='30%', height='30%', loc='center left', bbox_to_anchor=(0.5, 0.3, 0.8, 0.8), bbox_transform=ax.transAxes)
    axins.patch.set_edgecolor('black')  # Outline color
    axins.patch.set_linewidth(1)  

    for scenario_index, L_values_scenario in enumerate(L_values_list):
        summed_L_values = np.sum(compute_cumulative_packet_loss(L_values_scenario), axis=0)
        axins.plot(range(T), summed_L_values, linestyle='-', color=colors[scenario_index], linewidth=2.5)

    x1, x2 = T-2, T-1
    solid_lines_cumulatives = [np.sum(compute_cumulative_packet_loss(L_values_list[i]), axis=0) for i in [0,1,2,3,4]] 
    y1 = 0.99*min([np.min(cumulative[-1]) for cumulative in solid_lines_cumulatives])
    y2 = max([np.max(cumulative[-1]) for cumulative in solid_lines_cumulatives])

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticks([])
    axins.set_yticks([])

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    set_standard_plot_settings('Time step $t$', 'Cumulative Packet Loss', False, ax=ax)

    filename = generate_timestamped_filename('packet_loss_across_time')
    save_and_show_plot(filename)


def plot_mpc_packet_loss_across_W_values(mpc_results, W_values):
    
    colors, linestyles = get_plot_styles(len(W_values))
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    cumulative_costs_list = []  # List to store cumulative_costs

    for idx, (W, L_values_scenario) in enumerate(mpc_results.items()):
        cumulative_costs = compute_cumulative_sum(L_values_scenario)
        cumulative_costs_list.append(cumulative_costs)  # Append cumulative_costs to the list
        ax.plot(range(len(cumulative_costs)), cumulative_costs, linestyle=linestyles[idx % len(linestyles)], color=colors[idx], label=f'MPC, W={W}', linewidth=2.5)
    
    ax.set_yscale('log')
    set_standard_plot_settings('Time step $t$', 'Cumulative Packet Loss Cost', False)
    plt.legend()

    # Zoomed box
    axins = inset_axes(ax, width='30%', height='30%', loc='center left', bbox_to_anchor=(0.5, 0.3, 0.8, 0.8), bbox_transform=ax.transAxes)
    axins.patch.set_edgecolor('black')  
    axins.patch.set_linewidth(1)

    for idx, (W, L_values_scenario) in enumerate(mpc_results.items()):
        cumulative_costs = cumulative_costs_list[idx]  # Use the stored cumulative_costs
        axins.plot(range(len(cumulative_costs)), cumulative_costs, linestyle=linestyles[idx % len(linestyles)], color=colors[idx], label=f'MPC, W={W}', linewidth=2.5)

    # Zoom range for inset
    x1, x2 = T-2, T-1
    y1, y2 = get_zoomed_inset_ranges(cumulative_costs_list, x1, x2)  # Use the list of cumulative_costs
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")


    final_cumulative_costs = [compute_cumulative_sum(L_values_scenario, is_uncertainty= False)[-1] for L_values_scenario in mpc_results.values()]

    methods = [f'MPC, W={W}' for W in W_values]

    filename = generate_timestamped_filename(f'mpc_cumulative_cost_across_W_kp_{k_p}_M_{M}_P_{P}_Qbar_{Q_BAR}_T_{T}_dS_{DELTA_S}_dW_{DELTA_W_BAR}_NUMSIM_{NUM_SIM}')
    print_and_save_comparison_matrix(final_cumulative_costs, methods, filename, range_size = len(W_values))
    save_and_show_plot(filename)


def save_data_to_file(data, filename):
    print(filename)
    with open(filename, 'wb') as file:
        pickle.dump(data, file)