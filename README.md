# Online Interior Point Method

Critical analysis of the paper "Online Interior Point Methods for Time-Varying Equality Constraints"

## Repository Structure

### `analysis/` - Current Analysis (Correct)
Contains the **corrected and verified analysis** of the paper:
- `PAPER_ANALYSIS.md` - Detailed explanation of the paper's framework and claims
- `ANALYSIS_SUMMARY.md` - Summary of findings with correct understanding
- `lp_analysis.py` - Computational verification using linear programs

### `experiments/` - Exploratory Code
Experimental scripts used during the investigation:
- Various Python scripts exploring different aspects of the problem
- Generated plots and visualizations

### `old_analysis/` - Previous Analysis (Contains Errors)
**WARNING:** These files contain fundamental errors based on misunderstanding the paper's problem class.
They are kept for reference but should not be used.

Common errors in old analysis:
- Incorrectly assumed the paper was about pure linear programs
- Claimed Lagrangian Hessian is zero (wrong - it's the barrier Hessian)
- Misidentified the regret bound as O(√(V_T·T)) instead of O(V_T)

### `online-ipm/` - Implementation Code
Original implementation code and utilities.

## Key Findings

The paper is **mathematically sound** for its stated problem class (self-concordant barrier methods).

Main observations:
1. **Regret bound is very loose** - dominated by large constant term
2. **Constraint violation bound may be optimistic** - can be violated in practice with realistic adaptation rates
3. **Framework is theoretically valid** but practical value is limited

See `analysis/ANALYSIS_SUMMARY.md` for details.
