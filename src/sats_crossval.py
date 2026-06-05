import os
import numpy as np
from sklearn.linear_model import LinearRegression

LOGITS_DIR = r"C:\Users\Lenovo\Downloads\gen ai\results\logits"
RATIOS = ["0", "25", "50", "75", "100"]

# AG News target T (from Table 5 in paper)
T_AG_NEWS = [1.401, 1.478, 1.534, 1.534, 1.816]

# Calculate mu_z for AG News (using seed 42)
mu_z_ag = []
for r in RATIOS:
    logits = np.load(os.path.join(LOGITS_DIR, f"distilbert_s42_{r}_test_logits.npy"))
    mu = np.mean(np.abs(logits))
    mu_z_ag.append(mu)

# Fit SATS equation: T = alpha * mu_z + beta
X = np.array(mu_z_ag).reshape(-1, 1)
y = np.array(T_AG_NEWS)
reg = LinearRegression().fit(X, y)
alpha = reg.coef_[0]
beta = reg.intercept_

print(f"Fitted SATS on AG News: T = {alpha:.4f} * mu_z + {beta:.4f}")

# Calculate mu_z for SST-2 (using seed 42)
mu_z_sst2 = []
for r in RATIOS:
    logits = np.load(os.path.join(LOGITS_DIR, f"sst2_distilbert_s42_{r}_test_logits.npy"))
    mu = np.mean(np.abs(logits))
    mu_z_sst2.append(mu)

# Predict T for SST-2
T_pred_sst2 = reg.predict(np.array(mu_z_sst2).reshape(-1, 1))

# Actual T for SST-2 DistilBERT (from our new CSV)
T_actual_sst2 = [1.817, 1.876, 1.9665, 2.056, 2.2758]

print("\n--- Cross-Domain SATS Validation (SST-2) ---")
print(f"{'Ratio':<10} | {'Actual T':<10} | {'Predicted T':<12} | {'Error'}")
print("-" * 50)
for i, r in enumerate(RATIOS):
    actual = T_actual_sst2[i]
    pred = T_pred_sst2[i]
    err = pred - actual
    print(f"{r+'%':<10} | {actual:<10.3f} | {pred:<12.3f} | {err:+.3f}")
