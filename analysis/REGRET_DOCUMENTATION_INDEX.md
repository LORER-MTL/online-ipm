# Regret Derivation Documentation Index

## 📚 Complete Documentation for Regret Bound Derivation

This directory contains comprehensive documentation showing **step-by-step** how the regret bound is derived for OIPM with inequality constraints.

---

## 📖 Reading Order (Recommended)

### For Quick Understanding:
0. **`NOTATION_REFERENCE.md`** 📝 NOTATION GUIDE
   - Precise definitions of all symbols
   - What is $V_g^{\text{ineq}}$?
   - Common mistakes to avoid
   - Takes 15 minutes to read

1. **`REGRET_QUICK_REFERENCE.md`** ⭐ START HERE
   - 8-step derivation summary
   - All key equations in one place
   - Takes 10 minutes to read

### For Complete Details:
2. **`REGRET_DERIVATION_DETAILED.md`** 📊 COMPLETE DERIVATION
   - Full mathematical proof with all details
   - Step-by-step with explanations
   - Numerical example worked out
   - ~30 minutes to read thoroughly

### For Visual Learners:
3. **Visual Diagrams** 🎨
   - `vg_ineq_definition.png` - What exactly is $V_g^{\text{ineq}}$?
   - `regret_derivation_flowchart.png` - Complete 8-step flow
   - `regret_coupling_diagram.png` - Why (1 + ||F||) factor appears
   - `regret_bound_breakdown.png` - Numerical breakdown of terms

### For Implementation:
4. **`inequality_extension_demo.py`** 💻
   - Working Python code
   - Numerical verification of bounds
   - Generates additional visualizations

---

## 📄 File Descriptions

### Main Derivation Documents

#### `REGRET_QUICK_REFERENCE.md` (6 KB)
**What it is:** One-page cheat sheet with all key steps.

**Contains:**
- 8-step derivation outline
- All important equations boxed
- Table of term meanings
- Numerical example
- Common questions answered

**Best for:** Quick lookup, reviewing key steps, exam prep.

---

#### `REGRET_DERIVATION_DETAILED.md` (18 KB)
**What it is:** Complete rigorous mathematical derivation.

**Contains:**
- Detailed explanation of each step
- All intermediate calculations shown
- Mathematical tools explained
- Interpretation of results
- Numerical verification example
- Discussion of tightness

**Best for:** Understanding *why* each step works, self-study, research reference.

**Structure:**
- Step 1: Define regret for original problem
- Step 2: Transform to augmented problem
- Step 3: Prove regret equivalence ⭐
- Step 4: Apply Theorem 1
- Step 5: Decompose augmented path variation
- Step 6: Bound slack variation via coupling
- Step 7: Sum over time
- Step 8: Combine to get final bound

---

### Visualizations

#### `regret_derivation_flowchart.png` (342 KB)
**What it shows:** Complete logical flow from start to finish.

**Features:**
- Color-coded boxes for different types of steps
- Arrows showing dependencies
- Legend explaining color scheme
- All 8 steps in one view

**Best for:** Big picture understanding, presentations.

---

#### `regret_coupling_diagram.png` (285 KB)
**What it shows:** Why the (1 + ||F||) coupling factor appears.

**Features:**
- Visual comparison of original vs augmented space
- Shows how Δx induces Δs via coupling
- Geometric interpretation of bounds
- Formula derivation overlay

**Best for:** Understanding the most subtle part of the derivation.

---

#### `regret_bound_breakdown.png` (198 KB)
**What it shows:** Numerical breakdown of bound components.

**Features:**
- Left panel: Bound structure (constant + path terms)
- Right panel: Bar chart of term contributions
- Annotated with example parameter values
- Shows relative importance of each component

**Best for:** Understanding practical impact of different terms.

---

### Supporting Files

#### `inequality_extension.md` (11 KB)
Complete analysis of extending OIPM to inequalities.
- Problem transformation
- Theorem 1 application
- Guarantee transfer proof
- Includes regret derivation as part of broader analysis

#### `inequality_extension_demo.py` (17 KB)
Python implementation with:
- Slack transformation functions
- Regret bound computation
- Numerical verification
- Visualization generation

#### `regret_derivation_visualizer.py` (12 KB)
Python script that generated all three diagram files.
Run with: `python regret_derivation_visualizer.py`

---

## 🎯 Key Results at a Glance

### The Final Bound
$$R_d(T) \leq \frac{11p\beta}{5\eta_0(\beta-1)} + \|c\| \left[(1 + \|F\|) \cdot V_T^x + V_g^{\text{ineq}}\right]$$

### What Each Term Means

| Component | Meaning | Scales With |
|-----------|---------|-------------|
| $\frac{11p\beta}{5\eta_0(\beta-1)}$ | Barrier initialization | $O(p)$ - # inequalities |
| $\|c\| \cdot V_T^x$ | Direct tracking of $x$ | Path variation of optimal $x$ |
| $\|c\| \cdot \|F\| \cdot V_T^x$ | Coupling via slacks | Constraint matrix norm |
| $\|c\| \cdot V_g^{\text{ineq}}$ | Inequality RHS changes | $V_g^{\text{ineq}} = \sum_{t=1}^T \|g_t - g_{t-1}\|$ |

### Critical Insight
The regret in the **augmented problem** equals the regret in the **original problem** because slack variables have zero cost!

$$R_d^{\text{aug}}(T) = R_d(T)$$

This allows us to directly apply Theorem 1 to the augmented problem.

---

## 🔑 Key Mathematical Tools

The derivation uses these fundamental techniques:

1. **Triangle Inequality** (used 3 times)
   ```
   ||a + b|| ≤ ||a|| + ||b||
   ```

2. **Submultiplicativity of Matrix Norms**
   ```
   ||Mv|| ≤ ||M|| · ||v||
   ```

3. **Cost Equivalence**
   ```
   [c; 0]^T [x; s] = c^T x
   ```

4. **Theorem 1 from Paper** (black box application)
   ```
   R_d(T) ≤ (initialization) + ||c|| · (path variation)
   ```

---

## 📊 Numerical Example

From the demo with realistic problem size:

**Parameters:**
- n = 5 variables
- m = 2 equality constraints
- p = 3 inequality constraints
- T = 20 time steps
- ||c|| = 1.75, ||F|| = 2.67
- V_T^x = 1.06, V_g^ineq = 2.73
- β = 1.1, η₀ = 1.0

**Bound Components:**
- Constant term: 72.6
- Path term: 11.6
- **Total bound: 84.2**

**Verification:**
- Slack coupling: V_s = 3.13 ≤ 5.58 (bound) ✓
- Feasibility: All constraints satisfied ✓
- Bound holds: Actual regret < 84.2 ✓

---

## 🤔 Common Questions Answered

**Q: Why is the derivation so long?**
A: We need to carefully track how slack variables couple to original variables through the matrix F. The coupling factor (1 + ||F||) doesn't appear by accident!

**Q: Can I skip some steps?**
A: Steps 1-4 are straightforward. The subtle parts are Steps 6-7 (slack coupling). Focus there if short on time.

**Q: Is this bound tight?**
A: Yes! The (1 + ||F||) factor cannot be removed in general. We show examples where it's achieved.

**Q: What if I only care about static regret?**
A: This is dynamic regret (comparing to time-varying optimum). For static regret, replace V_T^x with smaller quantity.

**Q: What assumptions are needed?**
A: Same as paper: self-concordant barrier, bounded step changes, initial feasibility. See Section 2.1 in main paper.

---

## 🎓 Study Guide

### To Master the Derivation:

**Week 1:** Understand the transformation
- Read inequality_extension.md sections 1-2
- Work through slack variable examples by hand
- Verify that cost doesn't depend on slacks

**Week 2:** Follow the main proof
- Read REGRET_QUICK_REFERENCE.md completely
- For each step, verify one equation by hand
- Study the flowchart diagram

**Week 3:** Deep dive on coupling
- Read REGRET_DERIVATION_DETAILED.md sections 5-7
- Work through slack coupling example
- Understand why (1 + ||F||) appears

**Week 4:** Numerical verification
- Run inequality_extension_demo.py
- Modify parameters and see how bound changes
- Generate your own examples

---

## 🔬 For Researchers

### Using These Results:

**Citing this work:**
- Original paper: "Online Interior Point Methods for Time-Varying Equality Constraints"
- This extension: Slack variable transformation to handle inequalities

**Key theoretical contributions:**
1. Showed OIPM extends naturally to inequality constraints
2. Characterized coupling factor (1 + ||F||) precisely
3. Proved guarantees transfer with tight bounds

**Open questions:**
- Can initialization constant be improved?
- What about time-varying A and F matrices?
- Adaptive barrier parameter selection?

### Extending the analysis:
- See inequality_extension.md Section 8 for future directions
- Contact: Check repository for contributor info

---

## 📞 Getting Help

**If you're stuck on:**
- **Basic setup:** Read inequality_extension.md Section 1
- **Why regret equivalence works:** See REGRET_DERIVATION_DETAILED.md Step 3
- **The coupling factor:** Study regret_coupling_diagram.png
- **Numerical values:** Run inequality_extension_demo.py

**Still confused?**
Review in this order:
1. Quick reference → Get overview
2. Detailed derivation → Pick the confusing step
3. Visual diagram → See the big picture
4. Python demo → Verify with numbers

---

## 📦 Complete File Listing

```
analysis/
├── NOTATION_REFERENCE.md              # 📝 Precise definitions
├── REGRET_QUICK_REFERENCE.md          # ⭐ Start here
├── REGRET_DERIVATION_DETAILED.md      # Complete proof
├── regret_derivation_flowchart.png    # Visual flow
├── regret_coupling_diagram.png        # Coupling explanation
├── regret_bound_breakdown.png         # Numerical breakdown
├── inequality_extension.md            # Full analysis
├── inequality_extension_demo.py       # Python implementation
├── regret_derivation_visualizer.py   # Generates diagrams
├── INEQUALITY_EXTENSION_SUMMARY.md    # Executive summary
└── README_INEQUALITY_EXTENSION.md     # Complete guide
```

**Total documentation:** ~70 KB text + 825 KB images + 29 KB code

---

## ✅ Self-Check Questions

After studying, you should be able to:

1. ☐ Explain why R_d^aug(T) = R_d(T)
2. ☐ Write down the slack coupling inequality
3. ☐ Derive the (1 + ||F||) factor from scratch
4. ☐ Identify which terms come from initialization vs tracking
5. ☐ Compute the bound given problem parameters
6. ☐ Explain why the bound is tight
7. ☐ Implement slack transformation in code

---

**Last updated:** December 8, 2025

**Questions or suggestions?** Check the repository issues or pull request guidelines.
