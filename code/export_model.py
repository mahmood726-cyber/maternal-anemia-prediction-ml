import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Paths
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

def preprocess_data(df):
    df_clean = df.dropna(subset=["LBXHGB"]).copy()
    df_clean["anemia"] = (df_clean["LBXHGB"] < 11.0).astype(int)
    
    pir_median = df_clean["INDFMPIR"].median()
    df_clean["INDFMPIR"] = df_clean["INDFMPIR"].fillna(pir_median)
    
    df_clean["DMDEDUC2"] = df_clean["DMDEDUC2"].replace([7, 9, 7.0, 9.0], np.nan)
    edu_mode = df_clean["DMDEDUC2"].mode()[0]
    df_clean["DMDEDUC2"] = df_clean["DMDEDUC2"].fillna(edu_mode)
    
    df_clean["DMDMARTL"] = df_clean["DMDMARTL"].replace([77, 99, 77.0, 99.0], np.nan)
    marital_mode = df_clean["DMDMARTL"].mode()[0]
    df_clean["DMDMARTL"] = df_clean["DMDMARTL"].fillna(marital_mode)
    
    # Cast to int then str for one-hot encoding consistency
    df_clean["RIDRETH1"] = df_clean["RIDRETH1"].astype(int).astype(str)
    df_clean["DMDEDUC2"] = df_clean["DMDEDUC2"].astype(int).astype(str)
    df_clean["DMDMARTL"] = df_clean["DMDMARTL"].astype(int).astype(str)
    
    return df_clean

def export():
    data_path = os.path.join(PROCESSED_DIR, "nhanes_pregnant_raw.csv")
    df = pd.read_csv(data_path)
    df_clean = preprocess_data(df)
    
    feature_cols = ["RIDAGEYR", "RIDRETH1", "INDFMPIR", "DMDEDUC2", "DMDMARTL", "DMDFMSIZ"]
    X = df_clean[feature_cols].copy()
    y = df_clean["anemia"].values
    
    # Perform one-hot encoding
    # We specify columns explicitly to ensure we know the exact category mapping in JavaScript
    X_encoded = pd.get_dummies(X, columns=["RIDRETH1", "DMDEDUC2", "DMDMARTL"], drop_first=True)
    feature_names = X_encoded.columns.tolist()
    
    # Scale continuous variables
    # Continuous are: RIDAGEYR, INDFMPIR, DMDFMSIZ
    scaler = StandardScaler()
    X_matrix = X_encoded.values
    X_scaled = X_matrix.copy()
    X_scaled[:, :3] = scaler.fit_transform(X_matrix[:, :3])
    
    # Train Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_scaled, y)
    
    # Get coefficients
    coefs = lr.coef_[0].tolist()
    intercept = float(lr.intercept_[0])
    
    # Get scaling parameters
    means = scaler.mean_[:3].tolist()
    stds = scaler.scale_[:3].tolist()
    
    # Construct the model spec for export
    model_spec = {
        "model_type": "Logistic Regression (Balanced)",
        "features": feature_names,
        "coefficients": coefs,
        "intercept": intercept,
        "scaling": {
            "continuous_features": ["RIDAGEYR", "INDFMPIR", "DMDFMSIZ"],
            "means": means,
            "stds": stds
        },
        "categorical_mappings": {
            "RIDRETH1": {
                "1": "Mexican American",
                "2": "Other Hispanic",
                "3": "Non-Hispanic White",
                "4": "Non-Hispanic Black",
                "5": "Other Race (including Multi-Racial)"
            },
            "DMDEDUC2": {
                "1": "Less than 9th grade",
                "2": "9-11th grade (includes 12th grade no diploma)",
                "3": "High school graduate / GED",
                "4": "Some college or AA degree",
                "5": "College graduate or above"
            },
            "DMDMARTL": {
                "1": "Married",
                "2": "Widowed",
                "3": "Divorced",
                "4": "Separated",
                "5": "Never married",
                "6": "Living with partner"
            }
        }
    }
    
    export_path = os.path.join(RESULTS_DIR, "model_spec.json")
    with open(export_path, "w") as f:
        json.dump(model_spec, f, indent=4)
        
    print(f"Exported model spec successfully to {export_path}")
    print(f"Model intercept: {intercept:.4f}")
    for f_name, coef in zip(feature_names, coefs):
        print(f"Feature: {f_name:25} Coefficient: {coef: .4f}")

if __name__ == "__main__":
    export()
