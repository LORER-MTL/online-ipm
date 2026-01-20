"""Run all experiments demonstrating failure of inequality extensions.

This script runs both the slack variable projection and barrier reformulation
experiments, generating plots and summary statistics that demonstrate why
these approaches fail when extending OPEN-M to inequality constraints.
"""

from pathlib import Path


def main():
    """Run all experiments."""
    print("=" * 70)
    print("NUMERICAL EXPERIMENTS: FAILED INEQUALITY EXTENSIONS FOR OPEN-M")
    print("=" * 70)
    print()
    print("This demonstrates that naive extensions of OPEN-M to inequality")
    print("constraints fail. Two approaches are tested:")
    print()
    print("1. SLACK VARIABLE PROJECTION")
    print("   - Reformulate Fx <= g as Fx + s = g with s >= 0")
    print("   - Project onto augmented equality constraints")
    print("   - Clip s to s >= 0")
    print("   FAILURE: Clipping destroys equality Fx+s=g and can increase")
    print("   distance to optimum.")
    print()
    print("2. BARRIER REFORMULATION")
    print("   - Add log barrier: min c^T x + (1/mu) * sum(-log(g_i - F_i x))")
    print("   - Apply OPEN-M with barrier objective")
    print("   - Take full Newton step (no line search)")
    print("   FAILURE: Newton step exits feasible region, Hessian conditioning")
    print("   blows up near boundary, violating Lipschitz Hessian assumption.")
    print()
    print("-" * 70)
    print()

    # Import and run individual experiments
    from online_ipm.experiments import test_slack_projection
    from online_ipm.experiments import test_barrier_method

    test_slack_projection.main()
    print()
    test_barrier_method.main()

    # Final summary
    results_dir = Path(__file__).parent.parent / 'results'
    print()
    print("=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)
    print()
    print(f"Generated plots saved to: {results_dir.absolute()}")
    print()
    print("Files generated:")
    for f in sorted(results_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
