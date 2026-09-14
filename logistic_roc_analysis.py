import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

print("=== 1. DATA GENERATION & PREPARATION ===")

np.random.seed(42)
n = 300


x1 = np.random.normal(Loc=650, scale=50, size=n)
x2 = np.random.uniform(0.1, 0.8, size=n)


logit_p = -5.0 + 0.008 * x1 + 2.5 * x2
prob = 1 / (1 + np.exp(-logit_p))
y = np.random.binomial(1, prob, size=n)

df = pd.DataFrame({'CreditScore': x1, 'DebtRatio': x2})


X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)



print("\n=== 2. INFERENTIAL LOGISTIC REGRESSION (STATSMODELS) ===")

X_train_const = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_const).fit(disp=False)


odds_ratios = np.exp(logit_model.params)
p_values = logit_model.pvalues

results_df = pd.DataFrame({
    'Coefficient': logit_model.params,
    'Odds Ratio': odds_ratios,
    'p-value': p_values
})
print(results_df.round(4))



print("\n=== 3. PREDICTIVE EVALUATION & ROC-AUC ===")

clf = LogisticRegression()
clf.fit(X_train, y_train)


y_probs = clf.predict_proba(X_test)[:, 1]


auc_score = roc_auc_score(y_test, y_probs)

print(f"ROC-AUC Score: {auc_score:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, clf.predict(X_test), digits=3))
