import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm



print("=== 1. TIME SERIES ANALYSIS ===")

np.random.seed(42)
dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
trend = np.linspace(10, 50, 100)
seasonality = 5 * np.sin(np.linspace(0, 8 * np.pi, 100))
noise = np.random.normal(0, 2, 100)

ts_data = pd.Series(trend + seasonality + noise, index=dates)


rolling_mean = ts_data.rolling(window=7).mean()
ema = ts_data.ewm(span=7).mean()

print("Time Series Summary:")
print(f"Mean Value: {ts_data.mean():.2f}")
print(f"7-Day Rolling Mean (Latest): {rolling_mean.iloc[-1]:.2f}")
print(f"7-Day EMA (Latest): {ema.iloc[-1]:.2f}")


print("\n=== 2. A/B TESTING & HYPOTHESIS TESTING ===")


group_a = np.random.normal(loc=100, scale=15, size=200)
group_b = np.random.normal(loc=104, scale=15, size=200)


t_stat, p_val = stats.ttest_ind(group_a, group_b)


pooled_std = np.sqrt((np.var(group_a, ddof=1) + np.var(group_b, ddof=1)) / 2)
cohens_d = (np.mean(group_b) - np.mean(group_a)) / pooled_std

print(f"Group A Mean: {group_a.mean():.2f}")
print(f"Group B Mean: {group_b.mean():.2f}")
print(f"T-Statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
print(f"Cohen's d (Effect Size): {cohens_d:.4f}")

if p_val < 0.05:
    print("Result: Statistically significant difference between groups (p < 0.05).")
else:
    print("Result: No statistically significant difference detected.")
