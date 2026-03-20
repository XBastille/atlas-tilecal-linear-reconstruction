import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from src.io import load_shards, load_y_stats, denormalize

data_dir = "data"
y_stats = load_y_stats(data_dir)
train = load_shards(data_dir, "train")
val = load_shards(data_dir, "val")
X_tr, y_tr = train["X"], train["y"]
X_val, y_val = val["X"], val["y"]
X_hi_tr, y_hi_tr = train["X_hi"], train["y_hi"]
X_hi_val, y_hi_val = val["X_hi"], val["y_hi"]
y_val_phys = denormalize(y_val, y_stats["mean_lo"], y_stats["std_lo"])
mask = y_val_phys > 10.0

def eval_model(name, y_hat_val):
    y_hat_val_phys = denormalize(y_hat_val, y_stats["mean_lo"], y_stats["std_lo"])
    r = (y_hat_val_phys[mask] - y_val_phys[mask]) / y_val_phys[mask]
    rms = np.sqrt(np.mean(r**2))
    print(f"{name:35s}: RMS = {rms:.4f}")

print("--- REGULARIZATION GRID SEARCH ---")
eval_model("Base OLS", LinearRegression().fit(X_tr, y_tr).predict(X_val))
alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
best_lasso_rms = float('inf')
best_lasso_alpha = None

for a in alphas:
    model = Lasso(alpha=a, max_iter=10000)
    model.fit(X_tr, y_tr)
    y_hat_val = model.predict(X_val)
    y_hat_val_phys = denormalize(y_hat_val, y_stats["mean_lo"], y_stats["std_lo"])
    r = (y_hat_val_phys[mask] - y_val_phys[mask]) / y_val_phys[mask]
    rms = np.sqrt(np.mean(r**2))
    if rms < best_lasso_rms:
        best_lasso_rms = rms
        best_lasso_alpha = a

print(f"Best Lasso (alpha={best_lasso_alpha:1.0e}) : RMS = {best_lasso_rms:.4f}")

best_en_rms = float('inf')
best_en_alpha = None

for a in alphas:
    model = ElasticNet(alpha=a, max_iter=10000)
    model.fit(X_tr, y_tr)
    y_hat_val = model.predict(X_val)
    y_hat_val_phys = denormalize(y_hat_val, y_stats["mean_lo"], y_stats["std_lo"])
    r = (y_hat_val_phys[mask] - y_val_phys[mask]) / y_val_phys[mask]
    rms = np.sqrt(np.mean(r**2))
    if rms < best_en_rms:
        best_en_rms = rms
        best_en_alpha = a

print(f"Best ElasticNet (alpha={best_en_alpha:1.0e}) : RMS = {best_en_rms:.4f}")
print("\n--- NON-LINEAR FEATURE ENGINEERING ---")
poly = PolynomialFeatures(degree=2, include_bias=False)
scaler = StandardScaler()
X_tr_poly = poly.fit_transform(X_tr)
X_val_poly = poly.transform(X_val)
X_tr_poly_scaled = scaler.fit_transform(X_tr_poly)
X_val_poly_scaled = scaler.transform(X_val_poly)
eval_model("Quadratic w/ StandardScaler", LinearRegression().fit(X_tr_poly_scaled, y_tr).predict(X_val_poly_scaled))
print("\n--- CROSS-CHANNEL (HI-GAIN) INFO ---")
X_tr_both = np.concatenate([X_tr, X_hi_tr], axis=1)
X_val_both = np.concatenate([X_val, X_hi_val], axis=1)
X_tr_both_scaled = scaler.fit_transform(X_tr_both)
X_val_both_scaled = scaler.transform(X_val_both)
eval_model("OLS w/ X_lo+X_hi (Scaled)", LinearRegression().fit(X_tr_both_scaled, y_tr).predict(X_val_both_scaled))
