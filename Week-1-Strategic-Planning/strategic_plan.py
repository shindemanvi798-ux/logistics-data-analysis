import os
import pandas as pd

DATA_FILE = "DataCoSupplyChainDataset.csv"

if os.path.exists(DATA_FILE):
    data = pd.read_csv(DATA_FILE)

    print("First five records:")
    print(data.head())

    print("\nDataset information:")
    data.info()

    print("\nDescriptive statistics:")
    print(data.describe(include="all"))

    print("\nMissing values by column:")
    print(data.isnull().sum())

    print("\nDuplicate rows:", data.duplicated().sum())

else:
    print("Dataset not found.")
    print(f"Expected dataset file: {DATA_FILE}")
    print("The dataset will be added and processed in the subsequent analysis tasks.")

# Planned next steps:
# 1. Confirm relevant delivery, cost, distance and shipment columns.
# 2. Clean missing/inconsistent records.
# 3. Create delivery-time and delay features where supported.
# 4. Calculate logistics KPIs.
# 5. Perform EDA and visualization.
# 6. Build predictive models in later weeks.
