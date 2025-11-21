import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import joblib

# ---------------------------
# 1) Load dataset
# ---------------------------
df = pd.read_csv("Bank Customer Churn Prediction.csv")

# Drop customer_id if present
if "customer_id" in df.columns:
    df = df.drop("customer_id", axis=1)

# ---------------------------
# 2) Separate X and y
# ---------------------------
X = df.drop("churn", axis=1)
y = df["churn"]

# ---------------------------
# 3) Baseline accuracy
# ---------------------------
baseline_acc = y.value_counts(normalize=True).max()
print("Baseline Accuracy (majority class):", baseline_acc)

# ---------------------------
# 4) One-hot encode ONCE (for all data)
# ---------------------------
X = pd.get_dummies(X, drop_first=True)

# ---------------------------
# 5) Decide K so that:
#       - K <= 10
#       - each fold has at least 30 instances
# ---------------------------
n_samples = len(X)
max_splits_by_size = n_samples // 30      # each fold at least 30 samples

if max_splits_by_size < 2:
    raise ValueError(
        f"Not enough samples ({n_samples}) to create at least 2 folds "
        f"with >= 30 instances each."
    )

k = min(10, max_splits_by_size)
print(f"\nUsing StratifiedKFold with k = {k}")

# ---------------------------
# 6) K-Fold Cross-Validation
#    (scaling + SMOTE inside each fold)
# ---------------------------
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

fold_accuracies = []
fold_precisions = []
fold_recalls = []
fold_f1s = []

fold_num = 1

for train_idx, test_idx in skf.split(X, y):
    print(f"\n===== Fold {fold_num} / {k} =====")

    X_train_fold = X.iloc[train_idx]
    X_test_fold  = X.iloc[test_idx]
    y_train_fold = y.iloc[train_idx]
    y_test_fold  = y.iloc[test_idx]

    # Scale numeric features (here we scale the full feature set)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_test_scaled  = scaler.transform(X_test_fold)

    # SMOTE only on the training fold
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train_fold)

    print("Class distribution after SMOTE (this fold):")
    print(pd.Series(y_train_res).value_counts())

    # Train model on this fold
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train_res, y_train_res)

    # Evaluate on validation fold (no SMOTE on test)
    y_pred_fold = gb_model.predict(X_test_scaled)

    acc  = accuracy_score(y_test_fold, y_pred_fold)
    prec = precision_score(y_test_fold, y_pred_fold, zero_division=0)
    rec  = recall_score(y_test_fold, y_pred_fold, zero_division=0)
    f1   = f1_score(y_test_fold, y_pred_fold, zero_division=0)

    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)

    fold_accuracies.append(acc)
    fold_precisions.append(prec)
    fold_recalls.append(rec)
    fold_f1s.append(f1)

    fold_num += 1

# ---------------------------
# 7) Overall CV metrics
# ---------------------------
print("\n===== Cross-Validation Results (Mean ± Std) =====")
print(f"Accuracy : {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Precision: {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
print(f"Recall   : {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
print(f"F1 Score : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

# ---------------------------
# 8) Train FINAL model on full dataset
#    (after seeing CV performance)
# ---------------------------
print("\nTraining final model on FULL data...")

# Refit scaler on all data
final_scaler = StandardScaler()
X_scaled_full = final_scaler.fit_transform(X)

# SMOTE on full (scaled) data
final_smote = SMOTE(random_state=42)
X_res_full, y_res_full = final_smote.fit_resample(X_scaled_full, y)

print("\nClass distribution after SMOTE on full data:")
print(pd.Series(y_res_full).value_counts())

# Train final Gradient Boosting model
final_model = GradientBoostingClassifier(random_state=42)
final_model.fit(X_res_full, y_res_full)

# ---------------------------
# 9) Save model + scaler + columns
# ---------------------------
joblib.dump(final_model, "gradient_boosting_model.pkl")
joblib.dump(final_scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "train_columns.pkl")

print("\nFinal model, scaler, and columns saved successfully.")

# ---------------------------
# 10) Example: Evaluate on a random sample of REAL (unresampled) data
# ---------------------------
from sklearn.metrics import confusion_matrix

df_eval = pd.read_csv("Bank Customer Churn Prediction.csv")
if "customer_id" in df_eval.columns:
    df_eval = df_eval.drop("customer_id", axis=1)

sample = df_eval.sample(1000, random_state=1)

X_real = sample.drop("churn", axis=1)
y_real = sample["churn"]

X_real = pd.get_dummies(X_real, drop_first=True)
X_real = X_real.reindex(columns=X.columns, fill_value=0)
X_real_scaled = final_scaler.transform(X_real)

pred = final_model.predict(X_real_scaled)

print("\n=== Evaluation on 1000-sample holdout ===")
print("Accuracy on sample:", accuracy_score(y_real, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_real, pred))
print("\nClassification Report:\n", classification_report(y_real, pred))
