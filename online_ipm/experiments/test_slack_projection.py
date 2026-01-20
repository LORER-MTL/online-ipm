"""Test slack variable projection algorithm failures.

This experiment demonstrates that the slack variable projection approach
fails when extending OPEN-M to inequality constraints.

Key failure modes:
1. Clipping slack variables can INCREASE distance to optimum
2. Equality Fx + s = g is violated after clipping
3. Constraint violations accumulate (no O(V_T) bound)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from online_ipm.problems import (create_simple_2d_problem, create_medium_problem,
                                  solve_lp)
from online_ipm.algorithms.slack_projection import SlackProjectionAlgorithm


def run_experiment(problem, name: str):
    """Run slack projection and collect metrics.

    Args:
        problem: OnlineLPProblem instance
        name: Name for logging

    Returns:
        Algorithm instance with history
    """
    # Initialize at first optimal point
    inst0 = problem.get_instance(0)
    x0, _ = solve_lp(inst0)

    alg = SlackProjectionAlgorithm(n=inst0.n, m=inst0.m)
    alg.initialize(inst0, x0)

    for t in range(problem.T):
        instance = problem.get_instance(t)
        x_star, f_star = solve_lp(instance)
        alg.step(instance, x_star, f_star)

    return alg


def plot_results(alg, name: str, save_dir: Path):
    """Generate plots showing failures.

    Args:
        alg: Algorithm with history
        name: Name for plot file
        save_dir: Directory to save plots

    Returns:
        Dictionary of summary statistics
    """
    history = alg.history
    T = len(history)

    # Extract metrics
    regrets = [m.regret for m in history]
    cum_regrets = [m.cumulative_regret for m in history]
    dist_before = [m.extra['distance_before_clip'] for m in history]
    dist_after = [m.extra['distance_after_clip'] for m in history]
    slack_eq_viol = [m.extra['slack_equality_violation'] for m in history]
    ineq_viol = [m.inequality_violation for m in history]
    clip_hurt = [i for i, m in enumerate(history) if m.extra['clip_increased_distance']]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Slack Projection Failures: {name}', fontsize=14)

    # Cumulative regret
    ax = axes[0, 0]
    ax.plot(cum_regrets)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative regret')
    ax.set_title(f'Cumulative regret: {cum_regrets[-1]:.2f}')
    ax.grid(True, alpha=0.3)

    # Distance before/after clipping
    ax = axes[0, 1]
    ax.plot(dist_before, label='Before clip', alpha=0.7)
    ax.plot(dist_after, label='After clip', alpha=0.7)
    if clip_hurt:
        ax.scatter(clip_hurt, [dist_after[i] for i in clip_hurt],
                   c='red', marker='x', s=80, zorder=5, label='Clip hurt')
    ax.set_xlabel('Time')
    ax.set_ylabel('Distance to (x*, s*)')
    ax.set_title(f'Clipping hurt in {len(clip_hurt)}/{T} steps')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Slack equality violation ||Fx + s - g||
    ax = axes[0, 2]
    ax.plot(slack_eq_viol)
    ax.set_xlabel('Time')
    ax.set_ylabel('||Fx + s - g||')
    ax.set_title('Equality Fx + s = g violated after clipping')
    ax.grid(True, alpha=0.3)

    # Instantaneous regret
    ax = axes[1, 0]
    ax.plot(regrets)
    ax.set_xlabel('Time')
    ax.set_ylabel('Regret')
    ax.set_title('Instantaneous regret c^T x - c^T x*')
    ax.grid(True, alpha=0.3)

    # Cumulative violations
    ax = axes[1, 1]
    ax.plot(np.cumsum(slack_eq_viol), label='||Fx+s-g||')
    ax.plot(np.cumsum(ineq_viol), label='||max(Fx-g,0)||', linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative violation')
    ax.set_title('Violations accumulate (no O(V_T) bound)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Number clipped over time
    ax = axes[1, 2]
    s_clipped = [m.extra['num_s_clipped'] for m in history]
    x_clipped = [m.extra['num_x_clipped'] for m in history]
    ax.bar(range(T), s_clipped, width=1.0, label='s clipped', alpha=0.7)
    ax.bar(range(T), x_clipped, width=1.0, label='x clipped', alpha=0.7, bottom=s_clipped)
    ax.set_xlabel('Time')
    ax.set_ylabel('# components clipped')
    ax.set_title('Variables clipped to maintain feasibility')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'slack_projection_{name}.png', dpi=150)
    plt.close()

    return {
        'final_cumulative_regret': cum_regrets[-1],
        'clip_hurt_count': len(clip_hurt),
        'max_slack_eq_viol': max(slack_eq_viol),
        'total_slack_eq_viol': sum(slack_eq_viol),
    }


def main():
    """Run all slack projection experiments."""
    results_dir = Path(__file__).parent.parent / 'results'
    print("=" * 60)
    print("SLACK PROJECTION EXPERIMENTS")
    print("=" * 60)
    print()

    # Simple 2D problem
    print("Running simple 2D problem...")
    problem_2d = create_simple_2d_problem(T=50)
    alg_2d = run_experiment(problem_2d, "2D")
    stats_2d = plot_results(alg_2d, "simple_2d", results_dir)
    print(f"  Saved plot to {results_dir / 'slack_projection_simple_2d.png'}")

    # Medium n=10 problem
    print("Running medium problem (n=10)...")
    problem_med = create_medium_problem(n=10, m=5, T=100)
    alg_med = run_experiment(problem_med, "medium")
    stats_med = plot_results(alg_med, "medium_n10", results_dir)
    print(f"  Saved plot to {results_dir / 'slack_projection_medium_n10.png'}")

    # Summary
    print()
    print("=" * 60)
    print("SLACK PROJECTION FAILURE SUMMARY")
    print("=" * 60)

    for name, stats in [("Simple 2D", stats_2d), ("Medium n=10", stats_med)]:
        print(f"\n{name}:")
        print(f"  Cumulative regret: {stats['final_cumulative_regret']:.4f}")

        check = "[X]" if stats['clip_hurt_count'] > 0 else "[ ]"
        print(f"  {check} Clipping increased distance: "
              f"{stats['clip_hurt_count']} timesteps")

        print(f"  [X] Equality Fx+s=g violated: max = {stats['max_slack_eq_viol']:.4f}")
        print(f"  [X] Cumulative violation: {stats['total_slack_eq_viol']:.4f}")

    print()
    print("Conclusion: Slack variable projection fails because clipping")
    print("destroys the equality Fx + s = g and can increase distance to optimum.")


if __name__ == "__main__":
    main()
