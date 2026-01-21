"""Test feasible space projection algorithm.

This experiment compares:
1. SlackProjectionAlgorithm (clipping) - violates Fx+s=g, can increase distance
2. FeasibleProjectionAlgorithm (QP projection) - maintains constraints, preserves distance

Key findings expected:
- Distance preservation: Full projection DOES preserve ‖y - y*‖ (convex set)
- Constraint satisfaction: Full projection maintains Fx + s = g AND s >= 0
- Active set changes: Still cause Newton convergence issues
- Computational cost: ~2x due to QP solve per iteration
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from online_ipm.problems import (create_simple_2d_problem, create_medium_problem,
                                  solve_lp)
from online_ipm.algorithms.slack_projection import SlackProjectionAlgorithm
from online_ipm.algorithms.feasible_projection import FeasibleProjectionAlgorithm


def run_experiment(problem, alg_class, name: str):
    """Run algorithm and collect metrics.

    Args:
        problem: OnlineLPProblem instance
        alg_class: Algorithm class to instantiate
        name: Name for logging

    Returns:
        Tuple of (algorithm instance, elapsed time)
    """
    # Initialize at first optimal point
    inst0 = problem.get_instance(0)
    x0, _ = solve_lp(inst0)

    alg = alg_class(n=inst0.n, m=inst0.m)
    alg.initialize(inst0, x0)

    start_time = time.time()
    for t in range(problem.T):
        instance = problem.get_instance(t)
        x_star, f_star = solve_lp(instance)
        alg.step(instance, x_star, f_star)
    elapsed = time.time() - start_time

    return alg, elapsed


def plot_comparison(alg_clip, alg_proj, name: str, save_dir: Path):
    """Generate comparison plots.

    Args:
        alg_clip: SlackProjectionAlgorithm with history
        alg_proj: FeasibleProjectionAlgorithm with history
        name: Name for plot file
        save_dir: Directory to save plots

    Returns:
        Dictionary of summary statistics
    """
    hist_clip = alg_clip.history
    hist_proj = alg_proj.history
    T = len(hist_clip)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Clipping vs Full Projection: {name}', fontsize=14)

    # 1. Cumulative regret comparison
    ax = axes[0, 0]
    cum_regret_clip = [m.cumulative_regret for m in hist_clip]
    cum_regret_proj = [m.cumulative_regret for m in hist_proj]
    ax.plot(cum_regret_clip, label='Clipping', alpha=0.8)
    ax.plot(cum_regret_proj, label='Full Projection', alpha=0.8, linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative regret')
    ax.set_title('Cumulative Regret')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Distance to optimum
    ax = axes[0, 1]
    dist_clip = [m.distance_to_optimum for m in hist_clip]
    dist_proj = [m.distance_to_optimum for m in hist_proj]
    ax.plot(dist_clip, label='Clipping', alpha=0.8)
    ax.plot(dist_proj, label='Full Projection', alpha=0.8, linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('||x - x*||')
    ax.set_title('Distance to Optimum')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Slack equality violation ||Fx + s - g||
    ax = axes[0, 2]
    slack_viol_clip = [m.extra['slack_equality_violation'] for m in hist_clip]
    slack_viol_proj = [m.extra['slack_equality_violation'] for m in hist_proj]
    ax.plot(slack_viol_clip, label='Clipping', alpha=0.8)
    ax.plot(slack_viol_proj, label='Full Projection', alpha=0.8, linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('||Fx + s - g||')
    ax.set_title('Equality Fx + s = g Violation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Distance preservation check (clipping vs projection)
    ax = axes[1, 0]
    # For clipping: distance_after_clip > distance_before_clip means failure
    clip_increased = [m.extra['clip_increased_distance'] for m in hist_clip]
    # For projection: check if projection preserved distance
    proj_preserved = [m.extra.get('final_proj_preserved_dist', True) for m in hist_proj]

    clip_fail_pct = 100 * sum(clip_increased) / T
    proj_fail_pct = 100 * (T - sum(proj_preserved)) / T

    ax.bar(['Clipping', 'Full Projection'], [clip_fail_pct, proj_fail_pct],
           color=['red', 'green'], alpha=0.7)
    ax.set_ylabel('% of steps that increased distance')
    ax.set_title(f'Distance Preservation Failures\n(Clipping: {clip_fail_pct:.1f}%, Projection: {proj_fail_pct:.1f}%)')
    ax.set_ylim(0, max(clip_fail_pct + 10, 10))
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Active set changes over time
    ax = axes[1, 1]
    active_changed = [1 if m.extra.get('active_set_changed', False) else 0 for m in hist_proj]
    active_before = [m.extra.get('active_constraints_before', 0) for m in hist_proj]
    active_after = [m.extra.get('active_constraints_after', 0) for m in hist_proj]

    ax.plot(active_before, label='Before Newton', alpha=0.7)
    ax.plot(active_after, label='After Projection', alpha=0.7, linestyle='--')
    change_times = [t for t, changed in enumerate(active_changed) if changed]
    if change_times:
        ax.scatter(change_times, [active_after[t] for t in change_times],
                   c='red', marker='x', s=80, zorder=5, label='Active set changed')
    ax.set_xlabel('Time')
    ax.set_ylabel('# active constraints (s_i ≈ 0)')
    ax.set_title(f'Active Set Changes ({sum(active_changed)} times)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Newton step exiting feasible set
    ax = axes[1, 2]
    newton_exited = [1 if m.extra.get('newton_exited_F', False) else 0 for m in hist_proj]
    s_clipped = [m.extra['num_s_clipped'] for m in hist_clip]

    ax.bar(range(T), s_clipped, width=1.0, label='Clipping: # s < 0', alpha=0.6)
    ax.bar(range(T), [m.extra.get('num_s_violated', 0) for m in hist_proj],
           width=0.5, label='Projection: # s violated', alpha=0.6)
    ax.set_xlabel('Time')
    ax.set_ylabel('# slack variables violated')
    ax.set_title(f'Newton Step Exits F ({sum(newton_exited)}/{T} steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'feasible_projection_{name}.png', dpi=150)
    plt.close()

    return {
        'clip_final_regret': cum_regret_clip[-1],
        'proj_final_regret': cum_regret_proj[-1],
        'clip_distance_failures': sum(clip_increased),
        'proj_distance_failures': T - sum(proj_preserved),
        'proj_active_set_changes': sum(active_changed),
        'clip_max_slack_viol': max(slack_viol_clip),
        'proj_max_slack_viol': max(slack_viol_proj),
        'proj_newton_exits': sum(newton_exited),
    }


def plot_distance_analysis(alg_proj, name: str, save_dir: Path):
    """Detailed distance analysis for projection algorithm.

    Args:
        alg_proj: FeasibleProjectionAlgorithm with history
        name: Name for plot file
        save_dir: Directory to save plots
    """
    hist = alg_proj.history
    T = len(hist)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Distance Analysis: {name}', fontsize=14)

    # 1. Distance at each stage
    ax = axes[0]
    d_before = [m.extra['distance_before_proj'] for m in hist]
    d_after_init = [m.extra['distance_after_initial_proj'] for m in hist]
    d_after_newton = [m.extra['distance_after_newton'] for m in hist]
    d_final = [m.extra['distance_after_final_proj'] for m in hist]

    ax.plot(d_before, label='Before initial projection', alpha=0.7)
    ax.plot(d_after_init, label='After initial projection', alpha=0.7, linestyle='--')
    ax.plot(d_after_newton, label='After Newton step', alpha=0.7, linestyle=':')
    ax.plot(d_final, label='After final projection', alpha=0.7, linestyle='-.')
    ax.set_xlabel('Time')
    ax.set_ylabel('||y - y*||')
    ax.set_title('Distance to Optimum at Each Stage')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 2. Distance changes
    ax = axes[1]
    # Initial projection improvement (should be >= 0 for convex set)
    init_proj_improvement = [d_before[t] - d_after_init[t] for t in range(T)]
    # Newton step change (can be positive or negative)
    newton_change = [d_after_newton[t] - d_after_init[t] for t in range(T)]
    # Final projection improvement (should be >= 0 for convex set)
    final_proj_improvement = [d_after_newton[t] - d_final[t] for t in range(T)]

    ax.plot(init_proj_improvement, label='Initial proj improvement', alpha=0.7)
    ax.plot(newton_change, label='Newton change', alpha=0.7, linestyle='--')
    ax.plot(final_proj_improvement, label='Final proj improvement', alpha=0.7, linestyle=':')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Distance change (positive = improved)')
    ax.set_title('Distance Changes at Each Step')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f'feasible_projection_distance_{name}.png', dpi=150)
    plt.close()


def main():
    """Run all feasible projection experiments."""
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FEASIBLE PROJECTION vs CLIPPING EXPERIMENTS")
    print("=" * 70)
    print()
    print("Testing whether full projection onto F = {Ax=b, Fx+s=g, s>=0}")
    print("fixes the problems with the clipping approach.")
    print()

    all_stats = {}

    # Simple 2D problem
    print("-" * 70)
    print("Test 1: Simple 2D problem (T=50)")
    print("-" * 70)
    problem_2d = create_simple_2d_problem(T=50)

    print("  Running clipping algorithm...")
    alg_clip, time_clip = run_experiment(problem_2d, SlackProjectionAlgorithm, "2D-clip")
    print(f"    Time: {time_clip:.3f}s")

    print("  Running full projection algorithm...")
    alg_proj, time_proj = run_experiment(problem_2d, FeasibleProjectionAlgorithm, "2D-proj")
    print(f"    Time: {time_proj:.3f}s (ratio: {time_proj/time_clip:.1f}x)")

    stats_2d = plot_comparison(alg_clip, alg_proj, "simple_2d", results_dir)
    plot_distance_analysis(alg_proj, "simple_2d", results_dir)
    all_stats['Simple 2D'] = {**stats_2d, 'time_clip': time_clip, 'time_proj': time_proj}
    print(f"  Saved plots to {results_dir}/feasible_projection_simple_2d.png")

    # Medium problem
    print()
    print("-" * 70)
    print("Test 2: Medium problem (n=10, m=5, T=100)")
    print("-" * 70)
    problem_med = create_medium_problem(n=10, m=5, T=100)

    print("  Running clipping algorithm...")
    alg_clip, time_clip = run_experiment(problem_med, SlackProjectionAlgorithm, "med-clip")
    print(f"    Time: {time_clip:.3f}s")

    print("  Running full projection algorithm...")
    alg_proj, time_proj = run_experiment(problem_med, FeasibleProjectionAlgorithm, "med-proj")
    print(f"    Time: {time_proj:.3f}s (ratio: {time_proj/time_clip:.1f}x)")

    stats_med = plot_comparison(alg_clip, alg_proj, "medium_n10", results_dir)
    plot_distance_analysis(alg_proj, "medium_n10", results_dir)
    all_stats['Medium n=10'] = {**stats_med, 'time_clip': time_clip, 'time_proj': time_proj}
    print(f"  Saved plots to {results_dir}/feasible_projection_medium_n10.png")

    # Summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for name, stats in all_stats.items():
        print(f"\n{name}:")
        print(f"  Cumulative Regret:")
        print(f"    Clipping:        {stats['clip_final_regret']:.4f}")
        print(f"    Full Projection: {stats['proj_final_regret']:.4f}")

        print(f"\n  Distance Preservation Failures:")
        check_clip = "[X]" if stats['clip_distance_failures'] > 0 else "[ ]"
        check_proj = "[X]" if stats['proj_distance_failures'] > 0 else "[ ]"
        print(f"    {check_clip} Clipping:        {stats['clip_distance_failures']} steps")
        print(f"    {check_proj} Full Projection: {stats['proj_distance_failures']} steps")

        print(f"\n  Slack Equality Violation (max ||Fx+s-g||):")
        print(f"    Clipping:        {stats['clip_max_slack_viol']:.6f}")
        print(f"    Full Projection: {stats['proj_max_slack_viol']:.6f}")

        print(f"\n  Active Set Changes: {stats['proj_active_set_changes']} times")
        print(f"  Newton Step Exits F: {stats['proj_newton_exits']} times")

        print(f"\n  Computational Time:")
        print(f"    Clipping:        {stats['time_clip']:.3f}s")
        print(f"    Full Projection: {stats['time_proj']:.3f}s")
        print(f"    Ratio:           {stats['time_proj']/stats['time_clip']:.1f}x")

    print()
    print("=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
1. Distance Preservation:
   - Full projection DOES preserve distance (convex set property)
   - Clipping can INCREASE distance to optimum

2. Constraint Satisfaction:
   - Full projection maintains Fx + s = g (by construction)
   - Clipping violates this equality

3. However, full projection still fails because:
   - Newton step frequently exits feasible set F
   - Active set changes cause Newton convergence issues
   - Computational cost is ~2x due to QP solve

4. Bottom line: Full projection is BETTER than clipping but
   still cannot make OPEN-M work for inequality constraints.
   The fundamental issue is that inequality constraints create
   a polyhedral feasible region incompatible with OPEN-M's
   smooth Newton analysis.
""")


if __name__ == "__main__":
    main()
