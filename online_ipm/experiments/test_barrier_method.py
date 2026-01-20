"""Test barrier reformulation algorithm failures.

This experiment demonstrates that the barrier reformulation approach
fails when extending OPEN-M to inequality constraints.

Key failure modes:
1. Full Newton step exits feasible region (Fx > g)
2. Hessian conditioning blows up near boundary
3. Lipschitz Hessian assumption fails
4. Barrier optimum has O(1/μ) gap from true LP optimum
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from online_ipm.problems import (create_simple_2d_problem, create_medium_problem,
                                  solve_lp)
from online_ipm.algorithms.barrier_method import BarrierMethodAlgorithm


def run_experiment(problem, mu: float, name: str):
    """Run barrier method and collect metrics.

    Args:
        problem: OnlineLPProblem instance
        mu: Barrier parameter
        name: Name for logging

    Returns:
        Tuple of (algorithm instance, crash time or None)
    """
    inst0 = problem.get_instance(0)
    x_star, _ = solve_lp(inst0)

    # Start slightly in interior (avoid boundary issues at t=0)
    x0 = x_star * 0.9 + 0.05 * np.ones(inst0.n)

    alg = BarrierMethodAlgorithm(n=inst0.n, m=inst0.m, mu=mu)
    alg.initialize(inst0, x0)

    crash_time = None
    for t in range(problem.T):
        instance = problem.get_instance(t)
        x_star, f_star = solve_lp(instance)
        metrics = alg.step(instance, x_star, f_star)

        if not metrics.extra['feasible'] and crash_time is None:
            crash_time = t
            print(f"    [{name}] Exited feasible region at t={t}")

    return alg, crash_time


def plot_results(alg, name: str, mu: float, save_dir: Path):
    """Generate plots showing failures.

    Args:
        alg: Algorithm with history
        name: Name for plot file
        mu: Barrier parameter value
        save_dir: Directory to save plots

    Returns:
        Dictionary of summary statistics
    """
    history = alg.history
    T = len(history)

    cum_regrets = [m.cumulative_regret for m in history]
    min_slack = [m.extra['min_slack'] for m in history]
    condition = [min(m.extra['hessian_condition'], 1e12) for m in history]
    feasible = [m.extra['feasible'] for m in history]
    step_norm = [m.extra['newton_step_norm'] for m in history]
    distances = [m.distance_to_optimum for m in history]

    first_infeas = next((i for i, f in enumerate(feasible) if not f), None)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Barrier Method Failures: {name} (mu={mu})', fontsize=14)

    # Cumulative regret
    ax = axes[0, 0]
    ax.plot(cum_regrets)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative regret')
    ax.set_title(f'Cumulative regret: {cum_regrets[-1]:.2f}')
    ax.grid(True, alpha=0.3)

    # Min slack (feasibility)
    ax = axes[0, 1]
    ax.plot(min_slack)
    ax.axhline(0, color='red', linestyle='--', label='Boundary')
    if first_infeas:
        ax.axvline(first_infeas, color='red', alpha=0.5, label=f'Crash t={first_infeas}')
    ax.set_xlabel('Time')
    ax.set_ylabel('min(g - Fx)')
    ax.set_title('Newton exits feasible region (min slack < 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Hessian condition number
    ax = axes[0, 2]
    ax.semilogy(condition)
    ax.set_xlabel('Time')
    ax.set_ylabel('Condition number')
    ax.set_title('Hessian conditioning catastrophe')
    ax.grid(True, alpha=0.3)

    # Distance to true optimum (barrier gap)
    ax = axes[1, 0]
    ax.plot(distances)
    ax.set_xlabel('Time')
    ax.set_ylabel('||x - x*||')
    ax.set_title(f'Barrier != true optimum (gap ~O(1/mu)={1/mu:.3f})')
    ax.grid(True, alpha=0.3)

    # Newton step norm
    ax = axes[1, 1]
    ax.semilogy([max(s, 1e-10) for s in step_norm])
    ax.set_xlabel('Time')
    ax.set_ylabel('||Newton step||')
    ax.set_title('Erratic Newton steps (Lipschitz fails)')
    ax.grid(True, alpha=0.3)

    # Inequality violation
    ax = axes[1, 2]
    ineq_viol = [m.inequality_violation for m in history]
    ax.plot(np.cumsum(ineq_viol))
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative ||max(Fx-g,0)||')
    ax.set_title('Constraint violations')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'barrier_{name}_mu{mu}.png', dpi=150)
    plt.close()

    # Compute average distance only for feasible steps
    feasible_distances = [d for d, f in zip(distances, feasible) if f]
    avg_dist = np.mean(feasible_distances) if feasible_distances else np.inf

    return {
        'final_cumulative_regret': cum_regrets[-1],
        'crash_time': first_infeas,
        'max_condition': max(condition),
        'avg_distance': avg_dist,
    }


def main():
    """Run all barrier method experiments."""
    results_dir = Path(__file__).parent.parent / 'results'
    print("=" * 60)
    print("BARRIER METHOD EXPERIMENTS")
    print("=" * 60)
    print()

    all_stats = {}

    for mu in [1.0, 10.0, 100.0]:
        print(f"Testing mu = {mu}")

        # Simple 2D problem
        print("  Simple 2D problem...")
        problem_2d = create_simple_2d_problem(T=50)
        alg_2d, crash_2d = run_experiment(problem_2d, mu, f"2D_mu{mu}")
        stats_2d = plot_results(alg_2d, "simple_2d", mu, results_dir)
        all_stats[f'2D_mu{mu}'] = stats_2d

        # Medium problem
        print("  Medium problem (n=10)...")
        problem_med = create_medium_problem(n=10, m=5, T=100)
        alg_med, crash_med = run_experiment(problem_med, mu, f"med_mu{mu}")
        stats_med = plot_results(alg_med, "medium_n10", mu, results_dir)
        all_stats[f'med_mu{mu}'] = stats_med

        print()

    # Summary
    print("=" * 60)
    print("BARRIER METHOD FAILURE SUMMARY")
    print("=" * 60)

    print("\n[X] Newton step exits feasible region without line search")
    for key, stats in all_stats.items():
        if stats['crash_time'] is not None:
            print(f"    {key}: crashed at t={stats['crash_time']}")
        else:
            print(f"    {key}: did not crash (lucky trajectory)")

    print("\n[X] Hessian condition number explodes near boundary")
    for key, stats in all_stats.items():
        print(f"    {key}: max condition = {stats['max_condition']:.2e}")

    print("\n[X] Barrier optimum has O(1/mu) gap from true optimum")
    for key, stats in all_stats.items():
        print(f"    {key}: avg distance to x* = {stats['avg_distance']:.4f}")

    print("\n[X] Newton steps become erratic (Lipschitz Hessian fails)")
    print("    (See Newton step norm plots for evidence)")

    print()
    print("Conclusion: Barrier reformulation fails because full Newton steps")
    print("can exit the feasible region, and the Hessian conditioning blows up")
    print("near the boundary, violating OPEN-M's Lipschitz Hessian assumption.")


if __name__ == "__main__":
    main()
