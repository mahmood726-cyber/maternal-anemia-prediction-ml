import os
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

def preprocess_data(df):
    # 1. Drop rows missing the target variable (hemoglobin)
    df_clean = df.dropna(subset=["LBXHGB"]).copy()
    
    # 2. Create target (anemia = Hb < 11.0 g/dL for pregnant women)
    df_clean["anemia"] = (df_clean["LBXHGB"] < 11.0).astype(int)
    
    # 3. Clean features
    # Impute Poverty Income Ratio (INDFMPIR) with median
    pir_median = df_clean["INDFMPIR"].median()
    df_clean["INDFMPIR"] = df_clean["INDFMPIR"].fillna(pir_median)
    
    # For education (DMDEDUC2) and marital status (DMDMARTL), 
    # replace 'Refused' (7/77) or 'Don't know' (9/99) with NaN, then impute with mode.
    df_clean["DMDEDUC2"] = df_clean["DMDEDUC2"].replace([7, 9, 7.0, 9.0], np.nan)
    edu_mode = df_clean["DMDEDUC2"].mode()[0]
    df_clean["DMDEDUC2"] = df_clean["DMDEDUC2"].fillna(edu_mode)
    
    df_clean["DMDMARTL"] = df_clean["DMDMARTL"].replace([77, 99, 77.0, 99.0], np.nan)
    marital_mode = df_clean["DMDMARTL"].mode()[0]
    df_clean["DMDMARTL"] = df_clean["DMDMARTL"].fillna(marital_mode)
    
    # 4. Convert categories to object types for one-hot encoding
    df_clean["RIDRETH1"] = df_clean["RIDRETH1"].astype(int).astype(str)
    df_clean["DMDEDUC2"] = df_clean["DMDEDUC2"].astype(int).astype(str)
    df_clean["DMDMARTL"] = df_clean["DMDMARTL"].astype(int).astype(str)
    
    return df_clean

def train_and_evaluate():
    data_path = os.path.join(PROCESSED_DIR, "nhanes_pregnant_raw.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Clean data file not found at {data_path}")
        
    df = pd.read_csv(data_path)
    df_clean = preprocess_data(df)
    
    print(f"Original shape: {df.shape}, Preprocessed shape: {df_clean.shape}")
    print(f"Anemia rate: {df_clean['anemia'].mean():.2%} ({df_clean['anemia'].sum()} / {len(df_clean)})")
    
    # Define features
    feature_cols = ["RIDAGEYR", "RIDRETH1", "INDFMPIR", "DMDEDUC2", "DMDMARTL", "DMDFMSIZ"]
    X = df_clean[feature_cols].copy()
    y = df_clean["anemia"].values
    
    # One-hot encoding for categorical variables
    X_encoded = pd.get_dummies(X, columns=["RIDRETH1", "DMDEDUC2", "DMDMARTL"], drop_first=True)
    feature_names = X_encoded.columns.tolist()
    X_matrix = X_encoded.values
    
    # We will evaluate 3 models using Stratified 5-Fold CV
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(class_weight="balanced", n_estimators=100, max_depth=5, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Lists to store out-of-fold metrics
        oof_preds = np.zeros(len(y))
        oof_probs = np.zeros(len(y))
        
        accs, aucs, precs, recs, f1s = [], [], [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_matrix, y)):
            X_train, y_train = X_matrix[train_idx], y[train_idx]
            X_val, y_val = X_matrix[val_idx], y[val_idx]
            
            # Scale numeric features (Age, PIR, Family Size)
            # The indices for continuous features in X_encoded:
            # RIDAGEYR (col 0), INDFMPIR (col 1), DMDFMSIZ (col 2)
            scaler = StandardScaler()
            X_train_scaled = X_train.copy()
            X_val_scaled = X_val.copy()
            
            X_train_scaled[:, :3] = scaler.fit_transform(X_train[:, :3])
            X_val_scaled[:, :3] = scaler.transform(X_val[:, :3])
            
            # Fit model
            model.fit(X_train_scaled, y_train)
            
            # Predict
            preds = model.predict(X_val_scaled)
            probs = model.predict_proba(X_val_scaled)[:, 1]
            
            oof_preds[val_idx] = preds
            oof_probs[val_idx] = probs
            
            # Compute fold metrics
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
    
    # Save the best model (using Random Forest for demonstration)
    best_model_name = "Random Forest"
    best_model = models[best_model_name]
    
    # Fit best model on the entire dataset
    scaler = StandardScaler()
    X_full_scaled = X_matrix.copy()
    X_full_scaled[:, :3] = scaler.fit_transform(X_matrix[:, :3])
    best_model.fit(X_full_scaled, y)
    
    model_export_path = os.path.join(MODELS_DIR, "anemia_model_rf.pkl")
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
    plt.title("Out-of-Fold ROC Curves (5-Fold Stratified CV)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_plot_path = os.path.join(RESULTS_DIR, "roc_curves.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves plot to {roc_plot_path}")
    
    # 2. Random Forest Feature Importance
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 8))
    feat_imp.plot(kind='barh', color='skyblue')
    plt.title("Random Forest - Feature Importance (Full Model)")
    plt.xlabel("Relative Importance Score")
    plt.tight_layout()
    imp_plot_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(imp_plot_path, dpi=300)
    plt.close()
    print(f"Saved Feature Importance plot to {imp_plot_path}")
    
    # 3. Random Forest Confusion Matrix
    cm = confusion_matrix(y, results["Random Forest"]["oof_preds"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Anemic', 'Anemic'],
                yticklabels=['Non-Anemic', 'Anemic'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Random Forest - Out-of-Fold Confusion Matrix')
    plt.tight_layout()
    cm_plot_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix plot to {cm_plot_path}")
    
    # Write a summary text file
    summary_path = os.path.join(RESULTS_DIR, "model_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== Anemia Prediction ML Model Summary ===\n")
        f.write(f"Total Cohort Size (Clean): {len(df_clean)}\n")
        f.write(f"Anemic Cases: {df_clean['anemia'].sum()} ({df_clean['anemia'].mean():.2%})\n")
        f.write(f"Non-Anemic Cases: {len(df_clean) - df_clean['anemia'].sum()}\n\n")
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
    train_and_evaluate()
