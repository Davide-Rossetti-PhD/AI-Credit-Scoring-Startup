# ==========================================
# WETRUST CREDIT SCORING – FULL PIPELINE
# ==========================================
# Complete ML workflow for DemoDay:
# - data inspection & preprocessing
# - train / validation / test split
# - 5-fold cross-validation
# - model training (Logistic Regression + Random Forest baseline)
# - advanced evaluation (ROC, PRC, confusion matrix)
# - feature importance visualization
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, cross_validate
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    f1_score,
    accuracy_score
)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("crest")

# ==========================================
# 1. LOAD AND INSPECT DATA
# ==========================================
df = pd.read_csv("wetrust_synthetic_dataset.csv")
print("=== DATA OVERVIEW ===")
print(df.head(), "\n")
print("Shape:", df.shape)
print("Class distribution:\n", df["merit_class"].value_counts().sort_index(), "\n")

# Missing values check
print("Missing values per column:\n", df.isna().sum(), "\n")

# ==========================================
# 2. FEATURE / TARGET SPLIT
# ==========================================
X = df.drop(columns=["synthetic_score", "merit_class"])
y = df["merit_class"]

# Train / Validation / Test split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.20, stratify=y_temp, random_state=42
)
print(f"Train size: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}\n")

# ==========================================
# 3. SCALING (for models that need it)
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. DEFINE MODELS
# ==========================================
log_reg = LogisticRegression(solver="lbfgs", max_iter=5000, random_state=42)
rf = RandomForestClassifier(
    n_estimators=300, max_depth=6, random_state=42, class_weight="balanced"
)

models = {
    "Logistic Regression": (log_reg, X_train_scaled, X_val_scaled, X_test_scaled),
    "Random Forest": (rf, X_train, X_val, X_test)  # tree models don't need scaling
}

# ==========================================
# 5. CROSS-VALIDATION (5-fold)
# ==========================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, (mdl, Xtr, Xv, Xt) in models.items():
    cv_scores = cross_val_score(mdl, Xtr, y_train, cv=cv, scoring="accuracy")
    print(f"{name} – CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

print("\n")

# ==========================================
# 6. TRAIN ON TRAINING SET + VALIDATION EVAL
# ==========================================
results = []
for name, (mdl, Xtr, Xv, Xt) in models.items():
    mdl.fit(Xtr, y_train)
    y_val_pred = mdl.predict(Xv)
    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred, average="weighted")
    results.append({"Model": name, "Validation_Acc": acc, "Validation_F1": f1})

df_val = pd.DataFrame(results)
print("=== VALIDATION PERFORMANCE ===")
print(df_val, "\n")

# Pick best model based on validation F1
best_model_name = df_val.sort_values("Validation_F1", ascending=False).iloc[0]["Model"]
best_model, Xtr, Xv, Xt = models[best_model_name]
print(f"Best model selected: {best_model_name}\n")

# ==========================================
# 7. TEST SET EVALUATION
# ==========================================
best_model.fit(Xtr, y_train)
y_pred = best_model.predict(Xt)
y_prob = best_model.predict_proba(Xt)

# Define class labels from the dataset (1–5)
classes = sorted(y.unique())

print("=== CLASSIFICATION REPORT (TEST SET) ===")
print(classification_report(y_test, y_pred))

# Normalized Confusion Matrix (percentages)
cm_norm = confusion_matrix(y_test, y_pred, labels=classes, normalize="true")

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",              # display values as decimals
    cmap="Greens",          # consistent with your theme
    cbar=False,
    xticklabels=classes,
    yticklabels=classes
)
plt.title(f"Normalized Confusion Matrix – {best_model_name}")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.show()

# ==========================================
# 8. ROC-AUC & PRECISION-RECALL CURVES
# ==========================================
classes = sorted(y.unique())
y_test_bin = label_binarize(y_test, classes=classes)

roc_auc = roc_auc_score(y_test_bin, y_prob, average="macro")
print(f"Macro ROC-AUC: {roc_auc:.3f}\n")

# Plot ROC and PRC for each class
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, c in enumerate(classes):
    RocCurveDisplay.from_predictions(
        y_test_bin[:, i], y_prob[:, i], ax=axes[0], name=f"Class {c}"
    )
    PrecisionRecallDisplay.from_predictions(
        y_test_bin[:, i], y_prob[:, i], ax=axes[1], name=f"Class {c}"
    )

axes[0].set_title("ROC Curves – One-vs-Rest")
axes[1].set_title("Precision-Recall Curves – One-vs-Rest")
plt.tight_layout()
plt.show()

# ==========================================
# 9. FEATURE IMPORTANCE
# ==========================================
if best_model_name == "Logistic Regression":
    coef_mean = best_model.coef_.mean(axis=0)
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Mean_Coefficient": coef_mean
    }).sort_values("Mean_Coefficient", ascending=True)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=feature_importance,
        x="Mean_Coefficient",
        y="Feature",
        palette="coolwarm"
    )
    plt.title("Feature Importance – Logistic Regression Coefficients")
    plt.xlabel("Average Coefficient Value")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

else:  # Random Forest
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    }).sort_values("Importance", ascending=True)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=feature_importance,
        x="Importance",
        y="Feature",
        palette="viridis"
    )
    plt.title("Feature Importance – Random Forest")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

print("=== FEATURE IMPORTANCE (top features) ===")
print(feature_importance.sort_values(feature_importance.columns[-1], ascending=False).head(10), "\n")

# ==========================================
# 10. FINAL METRICS SUMMARY
# ==========================================
train_acc = accuracy_score(y_train, best_model.predict(Xtr))
test_acc = accuracy_score(y_test, y_pred)
print(f"Train Accuracy: {train_acc:.3f}")
print(f"Test Accuracy:  {test_acc:.3f}")
print(f"ROC-AUC (macro): {roc_auc:.3f}")
print("\nPipeline completed successfully 🎯")
