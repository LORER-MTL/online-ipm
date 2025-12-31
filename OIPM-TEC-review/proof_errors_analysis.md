# Analysis of Errors in Proofs

**Date:** December 30, 2025  
**Document:** main.tex

## Summary

This document analyzes critical errors found in multiple proofs in the paper. The errors range from incorrect matrix manipulations to unjustified inequalities that fundamentally undermine the mathematical validity of the proofs.

---

# Part 1: Errors in Lemma invHess (Lines 325-335)

## Overview

**Lemma invHess** claims:
```
‖D(y₁)D⁻¹(y₂)‖_{D(y₁)} ≤ 1/(1-‖y₁-y₂‖_{D(y₁)})²
```

This lemma is used throughout the paper, including in the proof of Lemma nred. However, **the proof is fundamentally flawed**.

---

## Error 1: False Equality (Line 328) - **CRITICAL**

### The Claim
```latex
‖D(y₁)D⁻¹(y₂)‖_{D(y₁)} = ‖(D(y₁)D⁻¹(y₂))⁻¹‖_{D(y₁)}
```

This claims that a matrix and its inverse have the **same Hessian-induced operator norm**.

### Why This is False

The notation ‖M‖_{D(y₁)} denotes the induced operator norm with respect to the Hessian norm ‖v‖_{D(y₁)} = √(vᵀD(y₁)v).

For any induced operator norm (including Hessian-induced norms), we have **‖M‖ · ‖M⁻¹‖ ≥ 1**, with equality only when M is an isometry in that norm.

**Counterexample:** If ‖M‖_{D(y₁)} = 2, then typically ‖M⁻¹‖_{D(y₁)} = 1/2 ≠ 2.

The equality **‖M‖_{D(y₁)} = ‖M⁻¹‖_{D(y₁)}** does NOT hold for general matrices, regardless of whether we use Euclidean or Hessian-induced norms.

### What This Means

The entire first line of the proof is **mathematically incorrect**. You cannot "invert the argument" and preserve the norm value.

---

## Error 1b: False Second Equality (Line 329) - **CRITICAL**

### The Claim
```latex
‖(D(y₁)D⁻¹(y₂))⁻¹‖_{D(y₁)} = min_{v≠0} ‖v‖²_{D(y₁)} / (vᵀD(y₂)v)
```

This claims that the induced operator norm of a matrix inverse (in the Hessian norm) equals a minimum generalized Rayleigh quotient.

### Notation Clarification

The notation ‖·‖_{D(y₁)} denotes the Hessian-induced norm:
- For vectors: ‖v‖_{D(y₁)} = √(vᵀD(y₁)v)
- For matrices: ‖M‖_{D(y₁)} is the induced operator norm with respect to this Hessian norm

### Why This is False

**Left side:** The Hessian-induced operator norm of M⁻¹ (where M = D(y₁)D⁻¹(y₂)):
```
‖M⁻¹‖_{D(y₁)} = sup_{v≠0} ‖M⁻¹v‖_{D(y₁)} / ‖v‖_{D(y₁)}
                = sup_{v≠0} √[(M⁻¹v)ᵀD(y₁)(M⁻¹v)] / √[vᵀD(y₁)v]
                = √[λ_max(M⁻ᵀD(y₁)M⁻¹, D(y₁))]
```

This is a **supremum** measuring how much the linear transformation M⁻¹ can stretch vectors in the D(y₁)-norm. It equals the square root of the maximum generalized eigenvalue.

**Right side:** The minimum generalized Rayleigh quotient:
```
min_{v≠0} (vᵀD(y₁)v) / (vᵀD(y₂)v) = λ_min(D(y₁), D(y₂))
```

This is the **minimum** eigenvalue of the matrix pencil, comparing two quadratic forms directly.

### The Fundamental Issue

Even though both sides involve the Hessian norm D(y₁), these are **completely different mathematical objects**:
- The operator norm ‖M⁻¹‖_{D(y₁)} involves **applying the linear transformation** M⁻¹ to vectors, then measuring their D(y₁)-norm
- The Rayleigh quotient min (vᵀD(y₁)v)/(vᵀD(y₂)v) is just a **ratio of two quadratic forms** without any matrix transformation applied to v

The equality cannot hold. This is a fundamental conceptual error, conflating induced operator norms (which involve matrix-vector products) with Rayleigh quotients (which are purely scalar ratios of quadratic forms).

---

## Error 2: Unjustified Transition from Min to Sup (Line 329-330)

### The Claim
```latex
min_{v≠0} ‖v‖²_{D(y₁)} / (vᵀD(y₁)D⁻¹(y₁)D(y₂)v) ≤ sup_{v≠0} ‖v‖²_{D(y₂)} / ‖v‖²_{D(y₁)}
```

### The Issue

Line 329 correctly simplifies: vᵀD(y₁)D⁻¹(y₁)D(y₂)v = vᵀD(y₂)v

So we have:
```
min_{v≠0} (vᵀD(y₁)v) / (vᵀD(y₂)v)
```

But then line 330 claims this relates to:
```
sup_{v≠0} (vᵀD(y₂)v) / (vᵀD(y₁)v)
```

**Problem:** The minimum of one quantity is **not directly related** to the supremum of its reciprocal without additional justification. For positive definite matrices:

```
min_{v≠0} A/B = 1/[max_{v≠0} B/A]
```

but this gives us **1/sup**, not ≤ sup.

---

## Error 3: Wrong Matrix Order

### What Boyd/Renegar Actually Provide

From the self-concordance definition (equation 4 in the paper):
```
‖v‖_{∇²f(x₂)} / ‖v‖_{∇²f(x₁)} ≤ 1/(1-‖x₂-x₁‖_{∇²f(x₁)})
```

Squaring both sides:
```
sup_{v≠0} (vᵀ∇²f(x₂)v) / (vᵀ∇²f(x₁)v) ≤ 1/(1-‖x₂-x₁‖_{∇²f(x₁)})²
```

This supremum equals **λ_max(H₁⁻¹H₂)** where H₁ = ∇²f(x₁), H₂ = ∇²f(x₂).

### The Correct Statement

For self-concordant functions, the standard result is:
```
λ_max(D(y₁)⁻¹D(y₂)) ≤ 1/(1-‖y₂-y₁‖_{D(y₁)})²
```

**But the lemma has D(y₁)D⁻¹(y₂)**, which is **D(y₁)D(y₂)⁻¹**, the reciprocal matrix!

This should give:
```
λ_max(D(y₁)D(y₂)⁻¹) = 1/λ_min(D(y₁)⁻¹D(y₂))
```

which is related to the **minimum** eigenvalue, not the maximum.

---

## Correct Statement (from Boyd/Renegar)

The standard self-concordance inequality gives:

**For the forward direction:**
```
‖v‖²_{D(y₂)} ≤ 1/(1-‖y₂-y₁‖_{D(y₁)})² · ‖v‖²_{D(y₁)}
```

**For the reverse direction:**
```
‖v‖²_{D(y₁)} ≤ 1/(1-‖y₁-y₂‖_{D(y₂)})² · ‖v‖²_{D(y₂)}
```

or equivalently:
```
‖v‖²_{D(y₁)} ≥ (1-‖y₁-y₂‖_{D(y₁)})² · ‖v‖²_{D(y₂)}
```

**References:**
- Boyd & Vandenberghe, "Convex Optimization," Section 9.6
- Renegar, "A Mathematical View of Interior-Point Methods," Chapter 2

---

## Impact on the Paper

This lemma (invHess) is used in:
1. Line 740 of the proof of Lemma nred
2. Potentially other places throughout the paper

**Since the lemma statement and proof are both incorrect, all results depending on it are suspect.**

---

## Recommendations for Lemma invHess

1. **Verify what inequality you actually need** for the rest of the paper
2. **Check if you need:**
   - λ_max(D(y₁)⁻¹D(y₂)) ≤ 1/(1-‖y₂-y₁‖_{D(y₁)})² (standard form), OR
   - Some other relation
3. **Rewrite the lemma** to match the correct standard result from Boyd/Renegar
4. **Fix all downstream proofs** that depend on this lemma

---

# Part 2: Errors in Proof of Lemma (nred) - Lines 719-751

## Summary

This proof contains **multiple critical errors** that undermine its mathematical validity. The claimed result may still be correct, but the proof as written has significant gaps and mistakes.

---

## Error 1: Unjustified Inequality (Line 726) - **CRITICAL**

### The Claim
```latex
r_t(y⁺,η)ᵀ D(y)⁻¹D(y)D(y⁺)⁻¹r_t(y⁺,η) 
  ≤ ‖D(y)D(y⁺)⁻¹‖_{D(y)} · r_t(y⁺,η)ᵀD(y)⁻¹r_t(y⁺,η)
```

### Why This is Problematic

Let r = r_t(y⁺,η) for brevity. The inequality states:

**LHS:** rᵀ D(y⁺)⁻¹ r = ‖r‖²_{D(y⁺)⁻¹}

**RHS:** ‖D(y)D(y⁺)⁻¹‖_{D(y)} · rᵀD(y)⁻¹r = ‖D(y)D(y⁺)⁻¹‖_{D(y)} · ‖r‖²_{D(y)⁻¹}

This would require:
```
‖r‖²_{D(y⁺)⁻¹} / ‖r‖²_{D(y)⁻¹} ≤ ‖D(y)D(y⁺)⁻¹‖_{D(y)}
```

### The Issue

From Lemma invHess (lines 321-323), the matrix norm is defined via:
```
‖M‖_{D(y)} = min_{v≠0} ‖v‖²_{D(y)} / (vᵀD(y)M⁻¹v)
```

or equivalently as an operator norm relating norms at different points.

However, the claimed inequality **conflates**:
1. The operator norm ‖D(y)D(y⁺)⁻¹‖_{D(y)} (which relates how norms change)
2. A specific quadratic form ratio for the vector r

The operator norm definition gives us:
```
‖v‖²_{D(y)} / ‖v‖²_{D(y⁺)} ≤ ‖D(y)D(y⁺)⁻¹‖²_{D(y)}
```

But we need the **inverse** relationship (with D⁻¹), which doesn't directly follow.

### Conclusion

**This step is unjustified** and potentially incorrect. The proof needs a different approach or much more careful justification of this inequality.

---

## Error 2: Notation Inconsistency (Line 738)

### The Problem
```latex
‖D(y)⁻¹[r_t(y⁺, η) - r_t(y, η)]‖
```

The variable suddenly switches from **η** (eta) to **η** in the subscript of r_t, when it should consistently be **η** (the barrier parameter used throughout).

### Fix
Should be:
```latex
‖D(y)⁻¹[r_t(y⁺, η) - r_t(y, η)]‖
```

---

## Error 3: Sign Error in Fundamental Theorem Application (Line 740)

### The Claim
```latex
∫₀¹ ‖D(y)⁻¹D(y - τn_t(y,η))‖ dτ
```

### The Problem

Since **y⁺ = y + n_t(y,η)**, applying the fundamental theorem of calculus to 
r_t(y⁺,η) - r_t(y,η) gives:

```
r_t(y⁺,η) - r_t(y,η) = ∫₀¹ D(y + τn_t(y,η)) · n_t(y,η) dτ
```

The integrand should involve **y + τn_t(y,η)**, NOT **y - τn_t(y,η)**.

### Fix
Should be:
```latex
∫₀¹ ‖D(y)⁻¹D(y + τn_t(y,η)) · n_t(y,η)‖ dτ
```

---

## Error 4: Missing Newton Step Factor (Line 740)

### The Problem

The integral in line 740 is missing the Newton step direction **n_t(y,η)** inside the norm:

**Current (incorrect):**
```latex
∫₀¹ ‖D(y)⁻¹D(y - τn_t(y,η))‖ dτ
```

**Should be:**
```latex
∫₀¹ ‖D(y)⁻¹D(y + τn_t(y,η)) · n_t(y,η)‖ dτ
```

### Why This Matters

Without the n_t(y,η) factor:
- The fundamental theorem of calculus is incorrectly applied
- You cannot factor out ‖n_t(y,η)‖ in the next step (line 744)
- The entire integral manipulation is invalid

The correct application should be:
```
D(y)⁻¹[r_t(y⁺,η) - r_t(y,η)] = D(y)⁻¹ ∫₀¹ D(y + τn_t(y,η)) · n_t(y,η) dτ
```

---

## Error 5: Incorrect Antiderivative (Lines 746-747)

### The Claim
```latex
∫₀¹ [a/(1-τa)²] dτ = [-1/(1-τa)]₀¹ + a
```

where a = ‖n_t(y,η)‖_{D(y)}.

### Verification by Differentiation

Check: d/dτ[-1/(1-τa)] = ?

```
d/dτ[-1/(1-τa)] = d/dτ[-(1-τa)⁻¹]
                 = -(-1)(1-τa)⁻²·(-a)
                 = -a/(1-τa)²
```

**This is the NEGATIVE of what we need!**

### Correct Antiderivative

The correct antiderivative of a/(1-τa)² is **+1/(1-τa)** (positive):

```
d/dτ[1/(1-τa)] = a/(1-τa)²  ✓
```

### Evaluation

**Correct calculation:**
```
∫₀¹ [a/(1-τa)²] dτ = [1/(1-τa)]₀¹ = 1/(1-a) - 1 = a/(1-a)
```

**What the proof claims:**
```
[-1/(1-τa)]₀¹ + a = -1/(1-a) - (-1) + a
                   = 1 - 1/(1-a) + a
                   = -a/(1-a) + a
                   = a²/(1-a)
```

### Interesting Note

Despite the **wrong antiderivative**, the final answer happens to be **algebraically correct** due to the extra +a term compensating for the sign error:

```
-a/(1-a) + a = [-a + a(1-a)]/(1-a) = a²/(1-a)  ✓
```

This suggests the authors may have **reverse-engineered** the calculation to get the desired result.

---

## Summary of All Errors

| Line | Error Type | Severity | Description |
|------|------------|----------|-------------|
| 726  | Unjustified inequality | **CRITICAL** | Matrix norm application is invalid or insufficiently justified |
| 738  | Notation inconsistency | Minor | η vs η mismatch |
| 740  | Sign error | Major | Should be y + τn, not y - τn |
| 740  | Missing factor | **CRITICAL** | Newton step n_t(y,η) missing from integrand |
| 746-747 | Wrong antiderivative | Major | Sign error that happens to cancel out |

---

## Recommended Actions

1. **Line 726:** Either find an alternative proof approach that doesn't use this matrix norm manipulation, or provide a rigorous justification for why this inequality holds.

2. **Line 738:** Fix notation consistency (η → η).

3. **Line 740:** Correct the sign (y - τn → y + τn) and include the missing Newton step factor.

4. **Lines 746-747:** Fix the antiderivative sign to use proper mathematical notation, even though the final result is accidentally correct.

5. **Overall:** Consider reviewing the proof against the original reference (likely Renegar's book) to ensure all steps are correctly adapted to the constrained setting.

---

## Impact Assessment

These errors suggest the proof has **not been carefully verified**. While the final result (Lemma nred) may still be correct, this proof as written:

- Contains **fundamental mathematical errors**
- Has **unjustified steps** that may not be valid
- Shows signs of **working backwards** from a desired result
- Cannot be trusted without major revision

**Recommendation:** Revise the entire proof or provide a reference to a correct proof in the literature.
