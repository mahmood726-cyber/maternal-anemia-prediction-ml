import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "processed"))
data_path = os.path.join(PROCESSED_DIR, "nhanes_pregnant_raw.csv")

if not os.path.exists(data_path):
    print(f"Error: {data_path} does not exist.")
    exit(1)

df = pd.read_csv(data_path)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

print("\nMissing values per column:")
print(df.isnull().sum())

# Define target variable (anemia = Hb < 11.0 g/dL for pregnant women)
# Filter out rows where Hemoglobin (LBXHGB) is missing
valid_df = df.dropna(subset=["LBXHGB"]).copy()
valid_df["anemia"] = (valid_df["LBXHGB"] < 11.0).astype(int)

print(f"\nCohort size after removing missing Hemoglobin: {len(valid_df)}")
anemia_counts = valid_df["anemia"].value_counts()
anemia_rate = valid_df["anemia"].mean()
print(f"Anemia status distribution:")
print(f"  Non-anemic (Hb >= 11.0): {anemia_counts.get(0, 0)}")
print(f"  Anemic (Hb < 11.0): {anemia_counts.get(1, 0)}")
print(f"  Anemia rate: {anemia_rate:.2%}")

print("\nSummary statistics for features:")
print(valid_df[["RIDAGEYR", "INDFMPIR", "DMDFMSIZ", "LBXHGB"]].describe())

print("\nRace/Ethnicity (RIDRETH3) distribution:")
print(valid_df["RIDRETH3"].value_counts(dropna=False))

print("\nEducation (DMDEDUC2) distribution:")
print(valid_df["DMDEDUC2"].value_counts(dropna=False))

print("\nMarital Status (DMDMARTL) distribution:")
print(valid_df["DMDMARTL"].value_counts(dropna=False))
