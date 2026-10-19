"""
Week 4: Predictive Modeling and Optimization in Logistics
Dataset: logistics_predictive_dataset.csv (hypothetical/simulated)
Target: Delivery_Time_Days
Run:
    pip install pandas numpy matplotlib scikit-learn
    python logistics_predictive_model.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "logistics_predictive_dataset.csv"
OUTPUT_PATH = BASE / "model_evaluation_results.csv"
VIS_DIR = BASE / "visualizations"
VIS_DIR.mkdir(exist_ok=True)

# 1. Load and validate the dataset
df = pd.read_csv(DATA_PATH)
required = ["Shipment_Distance_km", "Shipment_Volume_units", "Transportation_Mode",
            "Weather_Condition", "Traffic_Level", "Priority",
            "Estimated_Transportation_Cost_INR", "Delivery_Time_Days"]
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing required columns: {missing}")
df = df.drop_duplicates().copy()
df["Delivery_Time_Days"] = pd.to_numeric(df["Delivery_Time_Days"], errors="coerce")
df = df.dropna(subset=["Delivery_Time_Days"])

# 2. Separate features and target. Shipment_ID is excluded because it is only an identifier.
X = df.drop(columns=["Shipment_ID", "Delivery_Time_Days"], errors="ignore")
y = df["Delivery_Time_Days"]
categorical = ["Transportation_Mode", "Weather_Condition", "Traffic_Level", "Priority"]
numeric = [col for col in X.columns if col not in categorical]

# 3. Preprocess numeric and categorical columns inside a pipeline to avoid leakage.
preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 4. Train and compare a simple baseline with a nonlinear ensemble.
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
}
results = []
best_pipeline, best_rmse, best_name = None, float("inf"), None

for name, estimator in models.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Cross-validation is performed on the training set only.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    ).mean()
    results.append({
        "Model": name, "Test MAE (days)": mae, "Test RMSE (days)": rmse,
        "Test R-squared": r2, "5-fold CV MAE (days)": cv_mae
    })
    if rmse < best_rmse:
        best_pipeline, best_rmse, best_name = pipeline, rmse, name

results_df = pd.DataFrame(results).sort_values("Test RMSE (days)")
results_df.to_csv(OUTPUT_PATH, index=False)
print("\nModel evaluation:")
print(results_df.to_string(index=False))
print(f"\nSelected model based on lower test RMSE: {best_name}")

# 5. Refit the selected model on training data and produce diagnostic charts.
best_pipeline.fit(X_train, y_train)
test_predictions = best_pipeline.predict(X_test)

plt.figure(figsize=(7, 4))
plt.scatter(y_test, test_predictions, alpha=0.65)
limits = [min(y_test.min(), test_predictions.min()), max(y_test.max(), test_predictions.max())]
plt.plot(limits, limits, linestyle="--")
plt.xlabel("Actual delivery time (days)")
plt.ylabel("Predicted delivery time (days)")
plt.title(f"Actual vs Predicted Delivery Time: {best_name}")
plt.tight_layout()
plt.savefig(VIS_DIR / "actual_vs_predicted.png", dpi=160)
plt.close()

errors = y_test.to_numpy() - test_predictions
plt.figure(figsize=(7, 4))
plt.hist(errors, bins=20)
plt.xlabel("Prediction error (actual - predicted days)")
plt.ylabel("Shipment count")
plt.title("Prediction Error Distribution")
plt.tight_layout()
plt.savefig(VIS_DIR / "prediction_error_distribution.png", dpi=160)
plt.close()

print(f"\nSaved metrics to: {OUTPUT_PATH}")
print(f"Saved diagnostic charts to: {VIS_DIR}")
print("Note: results are based on hypothetical data, not real company shipments.")
