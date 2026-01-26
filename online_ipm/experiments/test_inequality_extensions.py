"""Compare inequality constraint extensions for OPEN-M.

This experiment compares two approaches for extending OPEN-M to handle
inequality constraints Fx <= g in addition to equality constraints Ax = b:

Approach 1: Infeasible-Start Newton with Clipping
    - After Newton step, clip x and s to ensure positivity
    - Relies on infeasible-start framework to correct violations

Approach 2: Primal-Dual IPM with Line Search
    - Use backtracking line search to maintain strict feasibility
    - More conservative but guaranteed to stay feasible

Both convert inequalities to slack form: Fx + s = g, s >= 0.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from online_ipm.problems import (create_simple_2d_problem, create_medium_problem,
                                  solve_lp)
from online_ipm.algorithms.infeasible_start_clipping import InfeasibleStartClippingAlgorithm
from online_ipm.algorithms.primal_dual_line_search import PrimalDualLineSearchAlgorithm


def run_comparison(problem, name: str, results_dir: Path):
    """Run both algorithms on a problem and compare.

    Args:
        problem: OnlineLPProblem instance
        name: Problem name for plots
        results_dir: Directory to save results

    Returns:
        Dictionary of summary statistics for both methods
    """
    inst0 = problem.get_instance(0)
    n, m = inst0.n, inst0.m

    # Get initial solution
    x_star_0, _ = solve_lp(inst0)
    # Start slightly in interior
    x0 = x_star_0 * 0.9 + 0.05 * np.ones(n)

    # Initialize both algorithms
    alg_clip = InfeasibleStartClippingAlgorithm(n=n, m=m, mu=1.0)
    alg_ls = PrimalDualLineSearchAlgorithm(n=n, m=m, mu=1.0)

    alg_clip.initialize(inst0, x0.copy())
    alg_ls.initialize(inst0, x0.copy())

    print(f"  Running {name} (T={problem.T}, n={n}, m={m})...")

    # Run through time steps
    for t in range(problem.T):
        instance = problem.get_instance(t)
        x_star, f_star = solve_lp(instance)

        alg_clip.step(instance, x_star, f_star)
        alg_ls.step(instance, x_star, f_star)

    # Generate comparison plots
    stats = plot_comparison(alg_clip, alg_ls, name, results_dir)

    return stats


def plot_comparison(alg_clip, alg_ls, name: str, save_dir: Path):
    """Generate comparison plots for both algorithms.

    Args:
        alg_clip: Infeasible-start clipping algorithm
        alg_ls: Primal-dual line search algorithm
        name: Name for plot files
        save_dir: Directory to save plots

    Returns:
        Dictionary of summary statistics
    """
    hist_clip = alg_clip.history
    hist_ls = alg_ls.history
    T = len(hist_clip)

    # Extract metrics
    cum_regret_clip = [m.cumulative_regret for m in hist_clip]
    cum_regret_ls = [m.cumulative_regret for m in hist_ls]

    dist_clip = [m.distance_to_optimum for m in hist_clip]
    dist_ls = [m.distance_to_optimum for m in hist_ls]

    eq_viol_clip = [m.equality_violation for m in hist_clip]
    eq_viol_ls = [m.equality_violation for m in hist_ls]

    ineq_viol_clip = [m.inequality_violation for m in hist_clip]
    ineq_viol_ls = [m.inequality_violation for m in hist_ls]

    # Algorithm-specific metrics
    num_clipped = [m.extra.get('num_clipped', 0) for m in hist_clip]
    alpha_primal = [m.extra.get('alpha_primal', 1.0) for m in hist_ls]

    min_x_clip = [m.extra.get('min_x', 0) for m in hist_clip]
    min_s_clip = [m.extra.get('min_s', 0) for m in hist_clip]
    min_x_ls = [m.extra.get('min_x', 0) for m in hist_ls]
    min_s_ls = [m.extra.get('min_s', 0) for m in hist_ls]

    r_slack_clip = [m.extra.get('r_slack_norm', 0) for m in hist_clip]
    r_slack_ls = [m.extra.get('r_slack_norm', 0) for m in hist_ls]

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Inequality Extension Comparison: {name}', fontsize=14)

    # Plot 1: Cumulative regret
    ax = axes[0, 0]
    ax.plot(cum_regret_clip, label='Clip+Infeasible', linewidth=2)
    ax.plot(cum_regret_ls, label='LineSearch', linewidth=2, linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title('Cumulative Regret')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Constraint violations (equality)
    ax = axes[0, 1]
    ax.plot(np.cumsum(eq_viol_clip), label='Clip+Infeasible', linewidth=2)
    ax.plot(np.cumsum(eq_viol_ls), label='LineSearch', linewidth=2, linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative ||Ax - b||')
    ax.set_title('Equality Constraint Violations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Constraint violations (inequality)
    ax = axes[0, 2]
    ax.plot(np.cumsum(ineq_viol_clip), label='Clip+Infeasible', linewidth=2)
    ax.plot(np.cumsum(ineq_viol_ls), label='LineSearch', linewidth=2, linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative ||max(Fx-g,0)||')
    ax.set_title('Inequality Constraint Violations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Clipping count vs step sizes
    ax = axes[1, 0]
    ax2 = ax.twinx()
    line1, = ax.plot(np.cumsum(num_clipped), 'b-', label='Clipping: cumulative clips', linewidth=2)
    line2, = ax2.plot(alpha_primal, 'r--', label='LineSearch: step size', linewidth=2, alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Clips', color='b')
    ax2.set_ylabel('Step Size α', color='r')
    ax.set_title('Clipping Count vs Step Sizes')
    ax.legend(handles=[line1, line2], loc='upper left')
    ax.grid(True, alpha=0.3)

    # Plot 5: Distance to optimum
    ax = axes[1, 1]
    ax.plot(dist_clip, label='Clip+Infeasible', linewidth=2)
    ax.plot(dist_ls, label='LineSearch', linewidth=2, linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('||x - x*||')
    ax.set_title('Distance to Optimum')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Slack constraint residual
    ax = axes[1, 2]
    ax.semilogy([max(r, 1e-12) for r in r_slack_clip], label='Clip+Infeasible', linewidth=2)
    ax.semilogy([max(r, 1e-12) for r in r_slack_ls], label='LineSearch', linewidth=2, linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('||Fx + s - g||')
    ax.set_title('Slack Constraint Residual (Fx + s = g)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'inequality_comparison_{name}.png', dpi=150)
    plt.close()

    # Compute summary statistics
    stats = {
        'clip': {
            'final_cumulative_regret': cum_regret_clip[-1],
            'avg_distance': np.mean(dist_clip),
            'total_clipped': sum(num_clipped),
            'avg_eq_violation': np.mean(eq_viol_clip),
            'avg_ineq_violation': np.mean(ineq_viol_clip),
            'avg_slack_residual': np.mean(r_slack_clip),
        },
        'line_search': {
            'final_cumulative_regret': cum_regret_ls[-1],
            'avg_distance': np.mean(dist_ls),
            'avg_step_size': np.mean(alpha_primal),
            'avg_eq_violation': np.mean(eq_viol_ls),
            'avg_ineq_violation': np.mean(ineq_viol_ls),
            'avg_slack_residual': np.mean(r_slack_ls),
        }
    }

    return stats


def print_summary(all_stats: dict):
    """Print a summary of results.

    Args:
        all_stats: Dictionary mapping problem names to statistics
    """
    print("\n" + "=" * 70)
    print("INEQUALITY EXTENSION COMPARISON SUMMARY")
    print("=" * 70)

    print("\n{:<20} {:>15} {:>15} {:>15}".format(
        "Problem", "Clip Regret", "LS Regret", "Winner"))
    print("-" * 70)

    for name, stats in all_stats.items():
        clip_regret = stats['clip']['final_cumulative_regret']
        ls_regret = stats['line_search']['final_cumulative_regret']
        winner = "Clip" if clip_regret < ls_regret else "LineSearch"
        print("{:<20} {:>15.2f} {:>15.2f} {:>15}".format(
            name, clip_regret, ls_regret, winner))

    print("\n" + "-" * 70)
    print("Constraint Violations:")
    print("{:<20} {:>20} {:>20}".format(
        "Problem", "Clip (avg ineq)", "LS (avg ineq)"))
    print("-" * 70)

    for name, stats in all_stats.items():
        clip_viol = stats['clip']['avg_ineq_violation']
        ls_viol = stats['line_search']['avg_ineq_violation']
        print("{:<20} {:>20.6f} {:>20.6f}".format(name, clip_viol, ls_viol))

    print("\n" + "-" * 70)
    print("Algorithm-Specific Metrics:")
    print("{:<20} {:>20} {:>20}".format(
        "Problem", "Total Clips", "Avg Step Size"))
    print("-" * 70)

    for name, stats in all_stats.items():
        total_clips = stats['clip']['total_clipped']
        avg_step = stats['line_search']['avg_step_size']
        print("{:<20} {:>20} {:>20.4f}".format(name, total_clips, avg_step))

    print("\n" + "=" * 70)
    print("OBSERVATIONS")
    print("=" * 70)

    print("""
1. Clipping Approach (Infeasible-Start):
   - Faster per iteration (no line search)
   - May introduce slack constraint violations (Fx + s = g)
   - Works well when constraints rarely binding

2. Line Search Approach:
   - More conservative step sizes
   - Better maintains feasibility
   - More stable in general

3. Key Trade-offs:
   - Speed vs. Stability
   - Constraint satisfaction vs. Progress
""")


def main():
    """Run all inequality extension experiments."""
    results_dir = Path(__file__).parent.parent / 'results'

    print("=" * 70)
    print("INEQUALITY EXTENSION EXPERIMENTS")
    print("Comparing: Clipping+Infeasible vs Primal-Dual Line Search")
    print("=" * 70)
    print()

    all_stats = {}

    # Simple 2D problem
    print("Problem 1: Simple 2D")
    problem_2d = create_simple_2d_problem(T=100)
    all_stats['simple_2d'] = run_comparison(problem_2d, 'simple_2d', results_dir)

    # Medium problem
    print("\nProblem 2: Medium (n=10, m=5)")
    problem_med = create_medium_problem(n=10, m=5, T=100)
    all_stats['medium_n10'] = run_comparison(problem_med, 'medium_n10', results_dir)

    # Larger problem
    print("\nProblem 3: Larger (n=20, m=10)")
    problem_large = create_medium_problem(n=20, m=10, T=100, seed=123)
    all_stats['larger_n20'] = run_comparison(problem_large, 'larger_n20', results_dir)

    # Print summary
    print_summary(all_stats)

    print(f"\nPlots saved to: {results_dir}/")


if __name__ == "__main__":
    main()
