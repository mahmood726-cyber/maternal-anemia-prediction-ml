import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve

# Paths
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def preprocess_nhanes(df):
    # Drop rows missing the target variable (hemoglobin)
    df_clean = df.dropna(subset=["LBXHGB"]).copy()
    df_clean["anemia"] = (df_clean["LBXHGB"] < 11.0).astype(int)
    
    # Impute Poverty Income Ratio (INDFMPIR) with median
    pir_median = df_clean["INDFMPIR"].median()
    df_clean["INDFMPIR"] = df_clean["INDFMPIR"].fillna(pir_median)
    
    # Replace Refused (7/77) or Don't know (9/99) with NaN, then impute with mode
    df_clean["DMDEDUC2"] = df_clean["DMDEDUC2"].replace([7, 9, 7.0, 9.0], np.nan)
    df_clean["DMDEDUC2"] = df_clean["DMDEDUC2"].fillna(df_clean["DMDEDUC2"].mode()[0])
    
    df_clean["DMDMARTL"] = df_clean["DMDMARTL"].replace([77, 99, 77.0, 99.0], np.nan)
    df_clean["DMDMARTL"] = df_clean["DMDMARTL"].fillna(df_clean["DMDMARTL"].mode()[0])
    
    # Cast variables for one-hot encoding
    df_clean["RIDRETH1"] = df_clean["RIDRETH1"].astype(int).astype(str)
    df_clean["DMDEDUC2"] = df_clean["DMDEDUC2"].astype(int).astype(str)
    df_clean["DMDMARTL"] = df_clean["DMDMARTL"].astype(int).astype(str)
    
    # Continuous: RIDAGEYR (0), INDFMPIR (1), DMDFMSIZ (2)
    feature_cols = ["RIDAGEYR", "RIDRETH1", "INDFMPIR", "DMDEDUC2", "DMDMARTL", "DMDFMSIZ"]
    X = df_clean[feature_cols].copy()
    X_encoded = pd.get_dummies(X, columns=["RIDRETH1", "DMDEDUC2", "DMDMARTL"], drop_first=True)
    
    return X_encoded, df_clean["anemia"].values

def preprocess_uganda_dhs(df):
    # Filter for currently pregnant women (v213 == 1) just in case
    df_clean = df[df["v213"] == 1].copy()
    
    # In raw DHS, v456 is Hemoglobin multiplied by 10
    # Drop missing values of hemoglobin (994, 995, 996, 999 are missing/refused in DHS codebook)
    df_clean = df_clean[df_clean["v456"].notnull()]
    df_clean = df_clean[(df_clean["v456"] < 250) & (df_clean["v456"] > 30)]
    
    # Target: Hb < 11.0 g/dL -> v456 < 110
    df_clean["anemia"] = (df_clean["v456"] < 110).astype(int)
    
    # Clean continuous features: v012 (Age), v190 (Wealth), v136 (Household size)
    # Fill any NaNs with median
    df_clean["v012"] = df_clean["v012"].fillna(df_clean["v012"].median())
    df_clean["v190"] = df_clean["v190"].fillna(df_clean["v190"].median())
    df_clean["v136"] = df_clean["v136"].fillna(df_clean["v136"].median())
    
    # Clean categorical features: v025 (Residence), v106 (Education), v113 (Water source), v501 (Marital status)
    # Replace Refused/Don't know in marital status (99 or 98) with NaN
    df_clean["v501"] = df_clean["v501"].replace([98, 99, 98.0, 99.0], np.nan)
    
    # Impute categorical NaNs with mode
    for col in ["v025", "v106", "v113", "v501"]:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            
        df_clean[col] = df_clean[col].astype(int).astype(str)
        
    # Columns to keep
    # Continuous are: v012, v190, v136
    feature_cols = ["v012", "v025", "v190", "v106", "v113", "v136", "v501"]
    X = df_clean[feature_cols].copy()
    
    # One-hot encode categoricals
    X_encoded = pd.get_dummies(X, columns=["v025", "v106", "v113", "v501"], drop_first=True)
    
    return X_encoded, df_clean["anemia"].values

def train_and_evaluate(dataset_name):
    if dataset_name == "nhanes":
        data_file = "nhanes_pregnant_raw.csv"
        prefix = ""
        model_name_suffix = "nhanes"
    else:
        data_file = "uganda_dhs_pregnant.csv"
        prefix = "uganda_"
        model_name_suffix = "uganda"
        
    data_path = os.path.join(PROCESSED_DIR, data_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Clean data file not found at {data_path}. Please run appropriate preprocessing script first.")
        
    df = pd.read_csv(data_path)
    
    if dataset_name == "nhanes":
        X_encoded, y = preprocess_nhanes(df)
    else:
        X_encoded, y = preprocess_uganda_dhs(df)
        
    feature_names = X_encoded.columns.tolist()
    X_matrix = X_encoded.values
    
    print(f"\n=================== Training on {dataset_name.upper()} ===================")
    print(f"Dataset shape: {X_encoded.shape}")
    print(f"Anemia rate: {y.mean():.2%} ({y.sum()} / {len(y)})")
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(class_weight="balanced", n_estimators=100, max_depth=5, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        oof_preds = np.zeros(len(y))
        oof_probs = np.zeros(len(y))
        
        accs, aucs, precs, recs, f1s = [], [], [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_matrix, y)):
            X_train, y_train = X_matrix[train_idx], y[train_idx]
            X_val, y_val = X_matrix[val_idx], y[val_idx]
            
            # Scale first 3 continuous columns
            scaler = StandardScaler()
            X_train_scaled = X_train.copy()
            X_val_scaled = X_val.copy()
            
            X_train_scaled[:, :3] = scaler.fit_transform(X_train[:, :3])
            X_val_scaled[:, :3] = scaler.transform(X_val[:, :3])
            
            model.fit(X_train_scaled, y_train)
            
            preds = model.predict(X_val_scaled)
            probs = model.predict_proba(X_val_scaled)[:, 1]
            
            oof_preds[val_idx] = preds
            oof_probs[val_idx] = probs
            
            accs.append(accuracy_score(y_val, preds))
            aucs.append(roc_auc_score(y_val, probs))
            precs.append(precision_score(y_val, preds, zero_division=0))
            recs.append(recall_score(y_val, preds))
            f1s.append(f1_score(y_val, preds, zero_division=0))
            
        print(f"  Accuracy:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"  ROC AUC:   {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
        print(f"  Precision: {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
        print(f"  Recall:    {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
        print(f"  F1-Score:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        
        results[model_name] = {
            "accuracy": np.mean(accs),
            "roc_auc": np.mean(aucs),
            "precision": np.mean(precs),
            "recall": np.mean(recs),
            "f1_score": np.mean(f1s),
            "oof_probs": oof_probs,
            "oof_preds": oof_preds
        }
        
    # Fit best model on the entire dataset
    # We will use Logistic Regression as the best primary/screening model
    best_model_name = "Logistic Regression"
    best_model = models[best_model_name]
    
    scaler = StandardScaler()
    X_full_scaled = X_matrix.copy()
    X_full_scaled[:, :3] = scaler.fit_transform(X_matrix[:, :3])
    best_model.fit(X_full_scaled, y)
    
    model_export_path = os.path.join(MODELS_DIR, f"anemia_model_lr_{model_name_suffix}.pkl")
    with open(model_export_path, "wb") as f:
        pickle.dump({
            "model": best_model,
            "scaler": scaler,
            "feature_names": feature_names
        }, f)
    print(f"\nSaved full {best_model_name} model to {model_export_path}")
    
    # ---------------- PLOTTING RESULTS ----------------
    sns.set_theme(style="whitegrid")
    
    # 1. ROC Curves
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y, res["oof_probs"])
        auc_val = roc_auc_score(y, res["oof_probs"])
        plt.plot(fpr, tpr, label=f"{name} (OOF AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{dataset_name.upper()} - Out-of-Fold ROC Curves (5-Fold Stratified CV)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_plot_path = os.path.join(RESULTS_DIR, f"{prefix}roc_curves.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves plot to {roc_plot_path}")
    
    # 2. Logistic Regression Coefficient / Feature Importance
    importances = np.abs(best_model.coef_[0])
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 8))
    feat_imp.plot(kind='barh', color='purple')
    plt.title(f"{dataset_name.upper()} - Feature Importance (Absolute Coefs)")
    plt.xlabel("Relative Importance Score (Absolute Coefficient)")
    plt.tight_layout()
    imp_plot_path = os.path.join(RESULTS_DIR, f"{prefix}feature_importance.png")
    plt.savefig(imp_plot_path, dpi=300)
    plt.close()
    print(f"Saved Feature Importance plot to {imp_plot_path}")
    
    # 3. Logistic Regression Confusion Matrix
    cm = confusion_matrix(y, results[best_model_name]["oof_preds"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
                xticklabels=['Non-Anemic', 'Anemic'],
                yticklabels=['Non-Anemic', 'Anemic'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f"{best_model_name} - {dataset_name.upper()} Confusion Matrix")
    plt.tight_layout()
    cm_plot_path = os.path.join(RESULTS_DIR, f"{prefix}confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix plot to {cm_plot_path}")
    
    # Write summary
    summary_path = os.path.join(RESULTS_DIR, f"{prefix}model_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"=== {dataset_name.upper()} Anemia Prediction ML Model Summary ===\n")
        f.write(f"Total Cohort Size (Clean): {len(y)}\n")
        f.write(f"Anemic Cases: {y.sum()} ({y.mean():.2%})\n")
        f.write(f"Non-Anemic Cases: {len(y) - y.sum()}\n\n")
        f.write("--- Out-of-Fold Performance (5-Fold Stratified CV) ---\n")
        for name, res in results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Accuracy:  {res['accuracy']:.4f}\n")
            f.write(f"  ROC AUC:   {res['roc_auc']:.4f}\n")
            f.write(f"  Precision: {res['precision']:.4f}\n")
            f.write(f"  Recall:    {res['recall']:.4f}\n")
            f.write(f"  F1-Score:  {res['f1_score']:.4f}\n")
    print(f"Saved performance summary text to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maternal Anemia ML Training Pipeline")
    parser.add_argument("--dataset", type=str, choices=["nhanes", "uganda"], default="nhanes",
                        help="Choose which dataset to train on: 'nhanes' or 'uganda'")
    args = parser.parse_args()
    
    train_and_evaluate(args.dataset)
