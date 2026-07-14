import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Paths
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

def preprocess_nhanes(df):
    df_clean = df.dropna(subset=["LBXHGB"]).copy()
    df_clean["anemia"] = (df_clean["LBXHGB"] < 11.0).astype(int)
    
    pir_median = df_clean["INDFMPIR"].median()
    df_clean["INDFMPIR"] = df_clean["INDFMPIR"].fillna(pir_median)
    
    df_clean["DMDEDUC2"] = df_clean["DMDEDUC2"].replace([7, 9, 7.0, 9.0], np.nan)
    df_clean["DMDEDUC2"] = df_clean["DMDEDUC2"].fillna(df_clean["DMDEDUC2"].mode()[0])
    
    df_clean["DMDMARTL"] = df_clean["DMDMARTL"].replace([77, 99, 77.0, 99.0], np.nan)
    df_clean["DMDMARTL"] = df_clean["DMDMARTL"].fillna(df_clean["DMDMARTL"].mode()[0])
    
    df_clean["RIDRETH1"] = df_clean["RIDRETH1"].astype(int).astype(str)
    df_clean["DMDEDUC2"] = df_clean["DMDEDUC2"].astype(int).astype(str)
    df_clean["DMDMARTL"] = df_clean["DMDMARTL"].astype(int).astype(str)
    
    feature_cols = ["RIDAGEYR", "RIDRETH1", "INDFMPIR", "DMDEDUC2", "DMDMARTL", "DMDFMSIZ"]
    X = df_clean[feature_cols].copy()
    X_encoded = pd.get_dummies(X, columns=["RIDRETH1", "DMDEDUC2", "DMDMARTL"], drop_first=True)
    return X_encoded, df_clean["anemia"].values

def preprocess_uganda(df):
    df_clean = df[df["v213"] == 1].copy()
    df_clean = df_clean[df_clean["v456"].notnull()]
    df_clean = df_clean[(df_clean["v456"] < 250) & (df_clean["v456"] > 30)]
    df_clean["anemia"] = (df_clean["v456"] < 110).astype(int)
    
    df_clean["v012"] = df_clean["v012"].fillna(df_clean["v012"].median())
    df_clean["v190"] = df_clean["v190"].fillna(df_clean["v190"].median())
    df_clean["v136"] = df_clean["v136"].fillna(df_clean["v136"].median())
    
    df_clean["v501"] = df_clean["v501"].replace([98, 99, 98.0, 99.0], np.nan)
    for col in ["v025", "v106", "v113", "v501"]:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        df_clean[col] = df_clean[col].astype(int).astype(str)
        
    feature_cols = ["v012", "v025", "v190", "v106", "v113", "v136", "v501"]
    X = df_clean[feature_cols].copy()
    X_encoded = pd.get_dummies(X, columns=["v025", "v106", "v113", "v501"], drop_first=True)
    return X_encoded, df_clean["anemia"].values

def export(dataset_name):
    if dataset_name == "nhanes":
        data_file = "nhanes_pregnant_raw.csv"
        export_file = "model_spec.json"
    else:
        data_file = "uganda_dhs_pregnant.csv"
        export_file = "uganda_model_spec.json"
        
    data_path = os.path.join(PROCESSED_DIR, data_file)
    df = pd.read_csv(data_path)
    
    if dataset_name == "nhanes":
        X_encoded, y = preprocess_nhanes(df)
        continuous_names = ["RIDAGEYR", "INDFMPIR", "DMDFMSIZ"]
        mappings = {
            "RIDRETH1": {"1": "Mexican American", "2": "Other Hispanic", "3": "Non-Hispanic White", "4": "Non-Hispanic Black", "5": "Other Race"},
            "DMDEDUC2": {"1": "Less than 9th grade", "2": "9-11th grade", "3": "High school grad", "4": "Some college", "5": "College grad"},
            "DMDMARTL": {"1": "Married", "2": "Widowed", "3": "Divorced", "4": "Separated", "5": "Never married", "6": "Living with partner"}
        }
    else:
        X_encoded, y = preprocess_uganda(df)
        continuous_names = ["v012", "v190", "v136"]
        mappings = {
            "v025": {"1": "Urban", "2": "Rural"},
            "v106": {"0": "No education", "1": "Primary", "2": "Secondary", "3": "Higher"},
            "v113": {"1": "Improved", "2": "Unimproved"},
            "v501": {"0": "Never married", "1": "Married", "2": "Cohabiting", "3": "Widowed", "4": "Divorced", "5": "Separated"}
        }
        
    feature_names = X_encoded.columns.tolist()
    X_matrix = X_encoded.values
    
    scaler = StandardScaler()
    X_scaled = X_matrix.copy()
    X_scaled[:, :3] = scaler.fit_transform(X_matrix[:, :3])
    
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_scaled, y)
    
    coefs = lr.coef_[0].tolist()
    intercept = float(lr.intercept_[0])
    means = scaler.mean_[:3].tolist()
    stds = scaler.scale_[:3].tolist()
    
    model_spec = {
        "model_type": f"Logistic Regression ({dataset_name.upper()} - Balanced)",
        "features": feature_names,
        "coefficients": coefs,
        "intercept": intercept,
        "scaling": {
            "continuous_features": continuous_names,
            "means": means,
            "stds": stds
        },
        "categorical_mappings": mappings
    }
    
    export_path = os.path.join(RESULTS_DIR, export_file)
    with open(export_path, "w") as f:
        json.dump(model_spec, f, indent=4)
        
    print(f"\n=================== Exported {dataset_name.upper()} Model Spec ===================")
    print(f"Exported to: {export_path}")
    print(f"Model intercept: {intercept:.4f}")
    for f_name, coef in zip(feature_names, coefs):
        print(f"Feature: {f_name:25} Coefficient: {coef: .4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Maternal Anemia LR Model Coefficients")
    parser.add_argument("--dataset", type=str, choices=["nhanes", "uganda"], default="nhanes",
                        help="Choose which model to export: 'nhanes' or 'uganda'")
    args = parser.parse_args()
    
    export(args.dataset)
