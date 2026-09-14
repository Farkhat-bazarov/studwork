import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler


np.random.seed(42)
n = 150

x1 = np.random.uniform(1, 10, n)
x2 = np.random.normal(5, 2, n)


errors = np.random.normal(0, 0.4 * x1)
y = 3.0 + 2.5 * x1 - 1.2 * x2 + errors

df = pd.DataFrame({'Experience': x1, 'Education': x2})


print("=== 1. ОЦЕНКА РЕГРЕССИИ (МНК / OLS) ===")


X_ols = sm.add_constant(df)
ols_model = sm.OLS(y, X_ols).fit()


bp_stat, bp_pvalue, _, _ = sms.het_breuschpagan(ols_model.resid, ols_model.model.exog)
print(f"Тест Бреуша-Пагана p-value: {bp_pvalue:.5f}")

if bp_pvalue < 0.05:
    print("-> Обнаружена гетероскедастичность! Считаем робастные ошибки (HC3):")
    # Пересчет со стандартными ошибками HC3
    robust_model = ols_model.get_robustcov_results(cov_type='HC3')
    print(robust_model.summary().tables[1])
else:
    print(ols_model.summary().tables[1])


print("\n=== 2. БУТСТРЕП ОЦЕНКА МЕДИАНЫ ===")

n_bootstraps = 2000
boot_medians = []

for _ in range(n_bootstraps):
    sample = np.random.choice(y, size=len(y), replace=True)
    boot_medians.append(np.median(sample))


ci_lower = np.percentile(boot_medians, 2.5)
ci_upper = np.percentile(boot_medians, 97.5)

print(f"Точечная медиана y: {np.median(y):.4f}")
print(f"95% Доверительный интервал (Bootstrap): [{ci_lower:.4f}, {ci_upper:.4f}]")


print("\n=== 3. РЕГУЛЯРИЗОВАННАЯ РЕГРЕССИЯ ===")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

ridge = Ridge(alpha=1.0).fit(X_scaled, y)
lasso = Lasso(alpha=0.1).fit(X_scaled, y)

print("Коэффициенты Ridge:", ridge.coef_)
print("Коэффициенты Lasso:", lasso.coef_)
