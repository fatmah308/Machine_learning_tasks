#importing files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Load and preprocess data
data = pd.read_csv("Covertype.csv")

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=["Soil_Type", "Wilderness_Area"])

X = data.drop("Cover_Type", axis=1)
y = data["Cover_Type"] - 1

# Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#Hyperparameter tuning for Random Forest

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf_model = RandomForestClassifier(random_state=42)

rf_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=rf_param_grid,
    n_iter=5,
    cv=3,
    n_jobs=1,  
    verbose=1,
    random_state=42
)

rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_

#Hyperparameter tuning for XGBoost
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_base = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, verbosity=0, random_state=42)

xgb_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=xgb_param_grid,
    n_iter=5,
    cv=3,
    n_jobs=1,  
    verbose=1,
    random_state=42
)

xgb_search.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=False
)

xgb_best = xgb_search.best_estimator_

#Evaluation function
def evaluate_model(model, name):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    f1 = report['macro avg']['f1-score']
    prec = report['macro avg']['precision']
    rec = report['macro avg']['recall']
    print(f"\n{name} Evaluation:")
    print(classification_report(y_test, preds))
    return [acc, prec, rec, f1]

# Evaluate both models
rf_metrics = evaluate_model(rf_best, "Random Forest")
xgb_metrics = evaluate_model(xgb_best, "XGBoost")

# Create comparison table
metrics_df = pd.DataFrame({
    "Random Forest": rf_metrics,
    "XGBoost": xgb_metrics
}, index=["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 Score (Macro)"])

print("\nModel Comparison Table:")
print(metrics_df.round(4))

# Confusion Matrices
def plot_conf_matrix(model, name):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

plot_conf_matrix(rf_best, "Random Forest")
plot_conf_matrix(xgb_best, "XGBoost")

# Step 6: Feature Importances
def plot_feature_importance(model, model_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:][::-1]
    plt.figure(figsize=(10,6))
    sns.barplot(x=importances[indices], y=np.array(X.columns)[indices])
    plt.title(f"Top 15 Features - {model_name}")
    plt.tight_layout()
    plt.show()

plot_feature_importance(rf_best, "Random Forest")
plot_feature_importance(xgb_best, "XGBoost")


