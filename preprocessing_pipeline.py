"""
Week 2: Logistics Data Cleaning and Preprocessing
Install dependencies: pip install pandas numpy scikit-learn
Put DataCoSupplyChainDataset.csv in this folder before running.

IMPORTANT:
- The public dataset's real column names must be verified before mapping
  them to the standardized fields used below.
- If the public CSV is absent, this script uses clearly labelled synthetic
  data for demonstration only. Synthetic values are not public-dataset results.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_FILE = Path("DataCoSupplyChainDataset.csv")
OUTPUT_FILE = Path("DataCoSupplyChainDataset_Preprocessed.csv")
OUTLIER_REPORT = Path("outlier_audit.csv")

if DATA_FILE.exists():
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded reference CSV: {DATA_FILE}")
    print("Source columns:", df.columns.tolist())
    print("Verify/map source columns before applying standardized field names.")
else:
    print("NOTICE: Reference CSV not found.")
    print("Using SYNTHETIC demonstration records; these are not DataCo findings.")
    df = pd.DataFrame({
        "Shipment_ID": ["S001", "S002", "S003", "S004", "S005",
                        "S006", "S006", "S008"],
        "Delivery_Time_Days": [3, 5, np.nan, 4, 25, 6, 6, 5],
        "Transportation_Cost": [1200, 1500, 1350, np.nan, 18000, 1600, 1600, 1550],
        "Shipment_Distance_km": [120, 250, 180, 210, 5000, 300, 300, 275],
        "Shipment_Quantity": [10, 20, 15, 18, 12, 22, 22, np.nan]
    })

print("\\n--- Initial profile ---")
print("Shape:", df.shape)
print(df.head())
print("\\nData types:\\n", df.dtypes)
print("\\nMissing values:\\n", df.isna().sum())
print("\\nExact duplicate rows:", int(df.duplicated().sum()))

clean = df.copy()

# These are standardized example names. Only columns actually present are processed.
numeric_candidates = [
    "Delivery_Time_Days", "Transportation_Cost",
    "Shipment_Distance_km", "Shipment_Quantity"
]
numeric_cols = [c for c in numeric_candidates if c in clean.columns]

for col in numeric_cols:
    clean[col] = pd.to_numeric(clean[col], errors="coerce")

duplicates_removed = int(clean.duplicated().sum())
clean = clean.drop_duplicates().copy()
print("\\nExact duplicate rows removed:", duplicates_removed)

missing_before = clean[numeric_cols].isna().sum() if numeric_cols else pd.Series(dtype=int)
print("\\nMissing numeric values before imputation:\\n", missing_before)

imputation_log = []
for col in numeric_cols:
    if clean[col].isna().any():
        median_value = clean[col].median()
        missing_count = int(clean[col].isna().sum())
        clean[col] = clean[col].fillna(median_value)
        imputation_log.append({
            "column": col, "missing_filled": missing_count,
            "median_used": float(median_value)
        })

print("\\nImputation audit:")
print(pd.DataFrame(imputation_log).to_string(index=False) if imputation_log else "No missing numeric values.")

# IQR audit. Potential outliers are capped for this demonstration, not automatically
# declared errors. Review genuine operational events before changing real records.
outlier_rows = []
for col in [c for c in ["Delivery_Time_Days", "Transportation_Cost",
                        "Shipment_Distance_km"] if c in clean.columns]:
    q1 = clean[col].quantile(0.25)
    q3 = clean[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (clean[col] < lower) | (clean[col] > upper)
    outlier_rows.append({
        "column": col, "Q1": q1, "Q3": q3, "IQR": iqr,
        "lower_bound": lower, "upper_bound": upper,
        "potential_outliers": int(mask.sum())
    })
    clean[col] = clean[col].clip(lower=lower, upper=upper)

audit = pd.DataFrame(outlier_rows)
print("\\nIQR audit:")
print(audit.to_string(index=False) if not audit.empty else "No mapped outlier columns.")
audit.to_csv(OUTLIER_REPORT, index=False)

# Keep a cleaned copy in original units for interpretation.
clean_original_units = clean.copy()

scale_cols = [c for c in numeric_candidates if c in clean.columns]
if scale_cols:
    scaler = MinMaxScaler()
    clean[scale_cols] = scaler.fit_transform(clean[scale_cols])

print("\\n--- Final checks ---")
print("Missing values:\\n", clean.isna().sum())
print("Exact duplicate rows:", int(clean.duplicated().sum()))
for col in scale_cols:
    print(f"{col}: scaled min={clean[col].min():.3f}, max={clean[col].max():.3f}")

clean.to_csv(OUTPUT_FILE, index=False)
clean_original_units.to_csv("DataCoSupplyChainDataset_Cleaned_Original_Units.csv", index=False)
print(f"Saved normalized dataset: {OUTPUT_FILE}")
print("Saved cleaned original-unit dataset: DataCoSupplyChainDataset_Cleaned_Original_Units.csv")
print(f"Saved IQR audit: {OUTLIER_REPORT}")
