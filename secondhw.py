"""
Statistical Tests as Linear Models — An Intuitive Tour
=======================================================
Inspired by: "Common statistical tests are linear models"
             by Jonas Kristoffer Lindelov

The big idea in plain English:
-------------------------------
Most statistical tests you've heard of — t-test, Mann-Whitney,
Wilcoxon, Wald — are really just asking ONE question dressed up
in different clothes:

    "Is there a relationship between x and y, or is it just noise?"

And they all answer it the same way under the hood:
by fitting a straight line  y = a + b*x
and checking whether the slope b is meaningfully different from zero.

That's it. Everything else is decoration.

What changes between tests:
    - What goes on the y-axis (raw values? ranks? differences?)
    - What goes on the x-axis (group labels? nothing? the data itself?)
    - How we decide "is b meaningfully different from zero?"

This script walks through four tests intuitively, shows you the
"classic" version and the "linear model" version side by side,
and confirms they give the same answer.

Requirements:
    pip install numpy scipy statsmodels matplotlib
"""

import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def compare(name_a, p_a, name_b, p_b):
    print(f"\n  Classic  ({name_a}):       p = {p_a:.4f}")
    print(f"  Linear model ({name_b}):  p = {p_b:.4f}")
    match = abs(p_a - p_b) < 0.01
    print(f"  {'✓ Same answer!' if match else '~ Close (small approx. error for ranked tests)'}")


# ──────────────────────────────────────────────────────────────
# TEST 1 — One-Sample Student's t-test
# ──────────────────────────────────────────────────────────────
banner("TEST 1 — One-Sample Student's t-test")

print("""
QUESTION: Is the average of my data different from some target value?

INTUITION:
  Imagine measuring coffee shop wait times.
  You claim "average wait = 5 minutes." Is that true?

  The t-test computes:
      (observed mean - target)
      divided by how noisy/spread the data is

  Big gap + small noise = probably real.
  Small gap + big noise = probably just chance.

AS A LINEAR MODEL:
  Draw a flat horizontal line through your data.
  That line IS the mean.
  The test asks: is that flat line sitting at 5, or shifted away?

  Model:  wait_time = intercept + noise
  We test whether the intercept equals our target.
  No x-variable needed — just a flat line (slope = 0).
""")

wait_times = np.array([4.2, 5.8, 3.9, 6.1, 4.7, 5.3, 4.5, 6.2, 3.8, 5.0,
                       4.9, 5.5, 4.1, 6.0, 4.4, 5.7, 3.7, 5.9, 4.6, 5.2])
target = 5.0

print(f"  Data: coffee wait times, n={len(wait_times)}")
print(f"  Sample mean = {wait_times.mean():.2f} minutes")
print(f"  We test: is the true mean = {target} minutes?")

_, p_classic = stats.ttest_1samp(wait_times, target)

# Shift data so the target becomes zero, then fit an intercept-only line
y_shifted = wait_times - target
X = np.ones(len(y_shifted))
lm = sm.OLS(y_shifted, X).fit()
p_lm = lm.pvalues[0]

compare("ttest_1samp", p_classic, "OLS intercept-only", p_lm)

print(f"\n  Interpretation: p = {p_classic:.4f}")
print(f"  → {'Mean differs significantly from 5 min.' if p_classic < 0.05 else 'No strong evidence mean differs from 5 min.'}")


# ──────────────────────────────────────────────────────────────
# TEST 2 — Two-Sample Student's t-test
# ──────────────────────────────────────────────────────────────
banner("TEST 2 — Two-Sample Student's t-test")

print("""
QUESTION: Do two groups have different averages?

INTUITION:
  You have Group A and Group B (e.g., two fertilizers).
  The t-test asks: "Is the gap between their means real, or just luck?"

  Big gap + small noise = probably real difference.
  Small gap + big noise = probably just chance.

AS A LINEAR MODEL:
  Label Group A as x=0 and Group B as x=1.
  Fit a line through all the data points.

  The SLOPE of that line = (mean of B) - (mean of A)

  If slope = 0, the groups are identical.
  If slope ≠ 0, there's a difference.

  Model:  height = intercept + slope * group_label + noise
  We test: is slope = 0?
""")

fertilizer_a = np.array([22, 19, 24, 21, 20, 23, 18, 25, 22, 21,
                          20, 24, 19, 23, 21, 22, 20, 25, 19, 23])
fertilizer_b = np.array([27, 30, 28, 31, 26, 29, 32, 27, 30, 28,
                          31, 29, 27, 30, 28, 32, 26, 29, 31, 28])

print(f"  Fertilizer A: mean = {fertilizer_a.mean():.1f} cm")
print(f"  Fertilizer B: mean = {fertilizer_b.mean():.1f} cm")
print(f"  Gap          = {fertilizer_b.mean() - fertilizer_a.mean():.1f} cm")

_, p_classic2 = stats.ttest_ind(fertilizer_a, fertilizer_b, equal_var=True)

y2 = np.concatenate([fertilizer_a, fertilizer_b])
x2 = np.array([0]*len(fertilizer_a) + [1]*len(fertilizer_b))
X2 = sm.add_constant(x2)
lm2 = sm.OLS(y2, X2).fit()
p_lm2 = lm2.pvalues[1]

compare("ttest_ind", p_classic2, "OLS slope on group label", p_lm2)

print(f"\n  Interpretation: p = {p_classic2:.4f}")
print(f"  → {'Fertilizer B produces significantly taller plants.' if p_classic2 < 0.05 else 'No significant difference between fertilizers.'}")


# ──────────────────────────────────────────────────────────────
# TEST 3 — Mann-Whitney U Test
# ──────────────────────────────────────────────────────────────
banner("TEST 3 — Mann-Whitney U Test")

print("""
QUESTION: Do two groups tend to have different values?
          (without assuming bell-shaped / normal data)

INTUITION:
  Same question as the t-test, but the data might be skewed or
  have outliers, so we don't trust raw averages.

  The trick: instead of comparing raw numbers, RANK everything
  from 1 (smallest) to N (largest), mixing both groups together.
  Then compare the average rank in Group A vs Group B.

  Why ranks? Because a value of 1,000,000 just becomes rank N.
  It can't skew things. Ranks are always evenly spaced.

AS A LINEAR MODEL:
  It's IDENTICAL to the two-sample t-test model —
  just run it on RANKS instead of raw values.

  Model:  rank(score) = intercept + slope * group_label + noise
  We test: is slope = 0?

  The p-values are close but not always exactly the same
  (ranks introduce a small approximation).
  With n > 30 per group they become virtually identical.
""")

satisfaction_a = np.array([3, 5, 4, 6, 3, 7, 4, 5, 3, 6,
                            4, 5, 3, 7, 4, 5, 6, 3, 4, 5])
satisfaction_b = np.array([6, 8, 7, 9, 6, 8, 7, 9, 8, 7,
                            6, 9, 7, 8, 6, 9, 7, 8, 9, 7])

print(f"  Store A: median = {np.median(satisfaction_a)}, mean = {satisfaction_a.mean():.1f}")
print(f"  Store B: median = {np.median(satisfaction_b)}, mean = {satisfaction_b.mean():.1f}")

_, p_mw = stats.mannwhitneyu(satisfaction_a, satisfaction_b, alternative='two-sided')

all_vals = np.concatenate([satisfaction_a, satisfaction_b])
x3 = np.array([0]*len(satisfaction_a) + [1]*len(satisfaction_b))
ranked_y = stats.rankdata(all_vals)   # <-- THE KEY STEP
X3 = sm.add_constant(x3)
lm3 = sm.OLS(ranked_y, X3).fit()
p_lm3 = lm3.pvalues[1]

compare("mannwhitneyu", p_mw, "OLS on ranks", p_lm3)

print(f"\n  Interpretation: p = {p_mw:.4f}")
print(f"  → {'Store B has significantly higher satisfaction.' if p_mw < 0.05 else 'No significant difference between stores.'}")
print("""
  NOTE: The p-values are close but not always exactly the same.
  The linear model on ranks is a valid approximation, and it
  becomes near-perfect for larger sample sizes (n > 30 per group).
""")


# ──────────────────────────────────────────────────────────────
# TEST 4 — Wilcoxon Signed-Rank Test
# ──────────────────────────────────────────────────────────────
banner("TEST 4 — Wilcoxon Signed-Rank Test")

print("""
QUESTION: In paired data (before/after), is there a consistent change?
          (without assuming bell-shaped differences)

INTUITION:
  You measure the same people before and after treatment.
  Compute: difference = after - before for each person.
  Now ask: do those differences consistently go in one direction?

  Like Mann-Whitney, we rank to avoid skew problems.
  But here we rank the SIZE of the differences (ignoring sign first),
  then put the signs back.

  If most of the BIGGEST differences are positive → strong upward shift.
  If big and small rank differences mix signs → probably just noise.

AS A LINEAR MODEL:
  Intercept-only model — but on SIGNED RANKS of the differences.

  Steps:
    1. Compute differences = after - before
    2. Rank their absolute sizes (biggest diff gets highest rank)
    3. Put the original sign back (was it positive or negative?)
    4. Fit a flat line through the signed ranks
    5. Is that flat line sitting at zero (no change) or shifted?

  Model:  signed_rank(diff) = intercept + noise
  H0: intercept = 0  (differences are centred at zero = no effect)
""")

bp_before = np.array([145, 138, 152, 141, 149, 136, 155, 143, 148, 140,
                       137, 153, 144, 150, 139, 146, 142, 151, 147, 138])
bp_after  = np.array([138, 130, 144, 133, 141, 128, 146, 135, 139, 132,
                       129, 145, 136, 142, 131, 138, 134, 143, 139, 130])
differences = bp_after - bp_before

print(f"  Participants: n={len(differences)}")
print(f"  Mean BP change (after - before): {differences.mean():.1f} mmHg")

_, p_wilcox = stats.wilcoxon(differences, alternative='two-sided')

abs_ranks   = stats.rankdata(np.abs(differences))
signed_ranks = abs_ranks * np.sign(differences)
X4 = np.ones(len(signed_ranks))
lm4 = sm.OLS(signed_ranks, X4).fit()
p_lm4 = lm4.pvalues[0]

compare("wilcoxon", p_wilcox, "OLS on signed ranks (intercept)", p_lm4)

print(f"\n  Interpretation: p = {p_wilcox:.4f}")
print(f"  → {'Medication significantly reduced blood pressure.' if p_wilcox < 0.05 else 'No significant effect of medication.'}")


# ──────────────────────────────────────────────────────────────
# TEST 5 — Wald Test
# ──────────────────────────────────────────────────────────────
banner("TEST 5 — Wald Test")

print("""
QUESTION: Is a specific coefficient in my model meaningfully non-zero?

INTUITION:
  After fitting a linear model, every coefficient has uncertainty.
  The Wald test asks a simple question:

      How many "uncertainty units" away from zero is this coefficient?

  coefficient / its standard error  =  Wald (t) statistic

  Large ratio → the coefficient is real, not just noise.
  Small ratio → could easily be zero by chance.

  Here's the punchline: all the t-tests above ARE Wald tests.
    - Test 1: Wald test on the intercept of a flat-line model
    - Test 2: Wald test on the slope (group predictor)
    - Test 3: Wald test on the slope, with y = ranks
    - Test 4: Wald test on the intercept, with y = signed ranks

  Same formula, every time. Different data transformations.
""")

hours_studied = np.array([1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
                           6, 7, 7, 8, 8, 9, 9, 10, 10, 11])
exam_score    = np.array([45, 50, 52, 55, 58, 60, 63, 65, 68, 70,
                           72, 74, 75, 78, 80, 82, 83, 85, 87, 90])

X5 = sm.add_constant(hours_studied)
lm5 = sm.OLS(exam_score, X5).fit()

slope     = lm5.params[1]
std_err   = lm5.bse[1]
wald_stat = slope / std_err
p_wald    = lm5.pvalues[1]

print(f"  Score per extra hour studied (slope): {slope:.2f}")
print(f"  Uncertainty of that slope (SE):       {std_err:.2f}")
print(f"  Wald statistic = slope / SE:          {wald_stat:.2f}")
print(f"  p-value:                              {p_wald:.6f}")
print(f"\n  Interpretation: p = {p_wald:.4f}")
print(f"  → {'Each extra study hour significantly raises scores.' if p_wald < 0.05 else 'No significant effect of study hours.'}")


# ──────────────────────────────────────────────────────────────
# SUMMARY FIGURE
# ──────────────────────────────────────────────────────────────
banner("VISUAL SUMMARY — All Tests as Linear Models")

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Statistical Tests as Linear Models — Intuitive Overview",
             fontsize=14, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Plot 1: One-sample t-test
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(range(len(wait_times)), wait_times, alpha=0.6, color='steelblue', s=30)
ax1.axhline(wait_times.mean(), color='steelblue', lw=2,
            label=f'Mean = {wait_times.mean():.2f}')
ax1.axhline(target, color='tomato', lw=2, linestyle='--',
            label=f'Target = {target}')
ax1.set_title("Test 1: One-Sample t-test\n(Is the flat line at the target?)", fontsize=9)
ax1.set_xlabel("Observation"); ax1.set_ylabel("Wait time (min)")
ax1.legend(fontsize=7)

# Plot 2: Two-sample t-test
ax2 = fig.add_subplot(gs[0, 1])
rng = np.random.default_rng(0)
jit = lambda n: rng.uniform(-0.1, 0.1, n)
ax2.scatter(jit(len(fertilizer_a)), fertilizer_a, alpha=0.6, color='steelblue', s=30, label='A')
ax2.scatter(1 + jit(len(fertilizer_b)), fertilizer_b, alpha=0.6, color='tomato', s=30, label='B')
ax2.plot([0, 1], lm2.params[0] + lm2.params[1]*np.array([0, 1]),
         'k-', lw=2, label='Slope = difference')
ax2.set_xticks([0, 1]); ax2.set_xticklabels(['A', 'B'])
ax2.set_title("Test 2: Two-Sample t-test\n(Is the slope = 0?)", fontsize=9)
ax2.set_xlabel("Group"); ax2.set_ylabel("Height (cm)")
ax2.legend(fontsize=7)

# Plot 3: Mann-Whitney — ranks
ax3 = fig.add_subplot(gs[0, 2])
ranked_a = ranked_y[:len(satisfaction_a)]
ranked_b = ranked_y[len(satisfaction_a):]
ax3.scatter(jit(len(ranked_a)), ranked_a, alpha=0.6, color='steelblue', s=30, label='Store A')
ax3.scatter(1 + jit(len(ranked_b)), ranked_b, alpha=0.6, color='tomato', s=30, label='Store B')
ax3.plot([0, 1], lm3.params[0] + lm3.params[1]*np.array([0, 1]),
         'k-', lw=2, label='Slope on ranks')
ax3.set_xticks([0, 1]); ax3.set_xticklabels(['Store A', 'Store B'])
ax3.set_title("Test 3: Mann-Whitney\n(Same as t-test but on RANKS)", fontsize=9)
ax3.set_xlabel("Store"); ax3.set_ylabel("Rank")
ax3.legend(fontsize=7)

# Plot 4: Wilcoxon signed ranks
ax4 = fig.add_subplot(gs[1, 0])
sorted_sr = np.sort(signed_ranks)
ax4.bar(range(len(sorted_sr)), sorted_sr,
        color=['tomato' if r < 0 else 'steelblue' for r in sorted_sr], alpha=0.7)
ax4.axhline(0, color='black', lw=1)
ax4.axhline(signed_ranks.mean(), color='purple', lw=2, linestyle='--',
            label=f'Mean signed rank = {signed_ranks.mean():.1f}')
ax4.set_title("Test 4: Wilcoxon Signed-Rank\n(Is mean of signed ranks = 0?)", fontsize=9)
ax4.set_xlabel("Participant (sorted)"); ax4.set_ylabel("Signed rank of BP diff")
ax4.legend(fontsize=7)

# Plot 5: Wald / regression
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(hours_studied, exam_score, alpha=0.7, color='steelblue', s=40, zorder=5)
x_fit = np.linspace(hours_studied.min(), hours_studied.max(), 100)
ax5.plot(x_fit, lm5.params[0] + lm5.params[1]*x_fit, 'tomato', lw=2,
         label=f'slope={lm5.params[1]:.1f}, p={p_wald:.4f}')
ax5.set_title("Test 5: Wald Test\n(Is the slope = 0?)", fontsize=9)
ax5.set_xlabel("Hours studied"); ax5.set_ylabel("Exam score")
ax5.legend(fontsize=7)

# Plot 6: Summary table
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
summary = (
    "THE UNIFYING IDEA\n"
    "─────────────────────────────────\n\n"
    "All tests fit:  y = a + b·x\n"
    "and ask:        is b = 0?\n\n"
    "What varies:\n\n"
    "  1-sample t\n"
    "    y = raw values\n"
    "    x = nothing (intercept only)\n\n"
    "  2-sample t\n"
    "    y = raw values\n"
    "    x = group label (0 or 1)\n\n"
    "  Mann-Whitney\n"
    "    y = RANKS of values\n"
    "    x = group label (0 or 1)\n\n"
    "  Wilcoxon\n"
    "    y = SIGNED RANKS of diffs\n"
    "    x = nothing (intercept only)\n\n"
    "  Wald\n"
    "    y = raw values\n"
    "    x = any continuous predictor"
)
ax6.text(0.02, 0.98, summary, transform=ax6.transAxes,
         fontsize=8.5, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig("/mnt/user-data/outputs/statistical_tests_summary.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved: statistical_tests_summary.png")


# ──────────────────────────────────────────────────────────────
# FINAL RECAP
# ──────────────────────────────────────────────────────────────
banner("FINAL RECAP — In Plain English")

print("""
  All four tests (and the Wald test) share the same skeleton:

    Fit a line:   y = intercept + slope * x + noise
    Ask:          is the slope (or intercept) meaningfully non-zero?

  ┌──────────────────┬──────────────────────┬──────────────────────┐
  │ Test             │ What is y?           │ What is x?           │
  ├──────────────────┼──────────────────────┼──────────────────────┤
  │ 1-sample t       │ Raw values           │ Nothing (intercept)  │
  │ 2-sample t       │ Raw values           │ Group label (0/1)    │
  │ Mann-Whitney     │ RANKS of values      │ Group label (0/1)    │
  │ Wilcoxon s-r     │ SIGNED RANKS of diff │ Nothing (intercept)  │
  │ Wald             │ Raw values           │ Any predictor        │
  └──────────────────┴──────────────────────┴──────────────────────┘

  Non-parametric tests (Mann-Whitney, Wilcoxon) are NOT magic.
  They just apply the same line-fitting to RANKS instead of raw
  numbers. Ranks can't be skewed. Ranks can't have huge outliers.
  That's the whole trick.

  The Wald test IS the t-test on any coefficient.
  Every p-value in a regression table is a Wald test.

  Lindelov's insight: one framework, endless special cases.
""")
