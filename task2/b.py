import numpy as np
import json
from scipy.optimize import curve_fit
from scipy.stats import chi2
from a import *

# Compute residual sum of squares for each model
rss_exp = np.sum((hist_cavity - exp_decay(bin_centers_c, *popt_v))**2)
rss_combined = np.sum((hist_cavity - combined_decay(bin_centers_c, *popt_c))**2)

# Likelihood Ratio Test (LRT)
df = len(popt_c) - len(popt_v)  # Degrees of freedom
lrt_stat = rss_exp - rss_combined  # Test statistic
p_value = 1 - chi2.cdf(lrt_stat, df)  # Compute p-value

# Decision at 95% confidence
reject_null = p_value < 0.05

# Output results
print("Best-fit parameters for Null Hypothesis (Exponential Decay in Cavity):", popt_v)
print("Best-fit parameters for Alternative Hypothesis (Exponential + Gaussian in Cavity):", popt_c)
print("Residual Sum of Squares for Null Hypothesis:", rss_exp)
print("Residual Sum of Squares for Alternative Hypothesis:", rss_combined)
print("Likelihood Ratio Test Statistic:", lrt_stat)
print("p-value:", p_value)
print("Reject Null Hypothesis:", reject_null)
