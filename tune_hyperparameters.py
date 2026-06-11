import os
import sys
import pickle
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CropCredit.Tuner")

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR   = "models"
RANDOM_STATE = 42
COMMODITIES  = ["Wheat", "Rice", "Potato", "Onion", "Tomato"]

FEATURE_COLUMNS = [
    "commodity_encoded", "tonnage", "market_arrivals",
    "current_price", "rainfall_deficit",
    "warehouse_temp", "humidity", "moisture_content",
]

def check_features(df):
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

def load_real_data(commodity):
    data_dir = "Data files"
    if not os.path.exists(data_dir):
        return None
    
    for filename in os.listdir(data_dir):
        if filename.lower() == f"cleaned_{commodity.lower()}_data.csv":
            filepath = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(filepath)
                required = FEATURE_COLUMNS + ["future_price", "spoilage_label"]
                missing = [c for c in required if c not in df.columns]
                if missing:
                    logger.warning(f"File {filename} is missing columns: {missing}")
                    return None
                return df
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
                return None
    return None

def tune_regressors(X_tr, y_tr, X_te, y_te, n_iter=10):
    logger.info(f"Tuning 3 Regressors for Financial Model (Price Prediction) with {n_iter} iterations ...")
    
    # 1. XGBoost Regressor
    xgb_base = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", eval_metric="mae")
    xgb_grid = {
        "n_estimators": [100, 200, 400],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [4, 6, 8],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
        "min_child_weight": [1, 3, 5]
    }
    
    # 2. Random Forest Regressor
    rf_base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    rf_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [6, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    }
    
    # 3. Gradient Boosting Regressor
    gb_base = GradientBoostingRegressor(random_state=RANDOM_STATE)
    gb_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.7, 0.8, 1.0]
    }
    
    algorithms = {
        "XGBoost Regressor": (xgb_base, xgb_grid),
        "Random Forest Regressor": (rf_base, rf_grid),
        "Gradient Boosting Regressor": (gb_base, gb_grid)
    }
    
    results = {}
    
    for name, (base, grid) in algorithms.items():
        logger.info(f"  > Hyperparameter searching for {name} ...")
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=grid,
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            cv=3,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        search.fit(X_tr[FEATURE_COLUMNS], y_tr)
        best_model = search.best_estimator_
        
        # Evaluate on test set
        preds = best_model.predict(X_te[FEATURE_COLUMNS])
        mae = mean_absolute_error(y_te, preds)
        r2 = r2_score(y_te, preds)
        
        results[name] = {
            "model": best_model,
            "params": search.best_params_,
            "val_mae": -search.best_score_,
            "test_mae": mae,
            "test_r2": r2
        }
    
    # Print comparison scorecard
    print("\n" + "=" * 80)
    print("  FINANCIAL RISK REGRESSION MODEL COMPARISON (Price Prediction)")
    print("=" * 80)
    print(f"  {'Algorithm':<30} | {'CV Val MAE':<12} | {'Test MAE':<10} | {'Test R²':<8}")
    print("-" * 80)
    for name, res in results.items():
        print(f"  {name:<30} | {res['val_mae']:<12.2f} | {res['test_mae']:<10.2f} | {res['test_r2']:<8.4f}")
    print("=" * 80)
    
    # Choose best algorithm based on lowest test MAE
    best_name = min(results, key=lambda k: results[k]["test_mae"])
    logger.info(f"Selected Best Financial Model: {best_name} (Test MAE: {results[best_name]['test_mae']:.2f})")
    
    return results[best_name]["model"], best_name, results[best_name]["params"]

def tune_classifiers(X_tr, y_tr, X_te, y_te, n_iter=10):
    logger.info(f"Tuning 3 Classifiers for Physical Model (Spoilage Prediction) with {n_iter} iterations ...")
    
    # 1. Random Forest Classifier
    rf_base = RandomForestClassifier(class_weight="balanced", oob_score=True, random_state=RANDOM_STATE, n_jobs=-1)
    rf_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [6, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    }
    
    # 2. XGBoost Classifier
    xgb_base = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss")
    xgb_grid = {
        "n_estimators": [100, 200, 400],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [4, 6, 8],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
        "min_child_weight": [1, 3, 5]
    }
    
    # 3. Gradient Boosting Classifier
    gb_base = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gb_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.7, 0.8, 1.0]
    }
    
    # Check if single-class target is present in training data
    unique_classes = np.unique(y_tr)
    if len(unique_classes) < 2:
        logger.warning("Only one target class present in training split. Skipping CV tuning and using default fitted Random Forest Classifier.")
        rf_base.fit(X_tr[FEATURE_COLUMNS], y_tr)
        return rf_base, "Default Random Forest Classifier (Single-Class Fallback)", {}
        
    algorithms = {
        "Random Forest Classifier": (rf_base, rf_grid),
        "XGBoost Classifier": (xgb_base, xgb_grid),
        "Gradient Boosting Classifier": (gb_base, gb_grid)
    }
    
    results = {}
    
    for name, (base, grid) in algorithms.items():
        logger.info(f"  > Hyperparameter searching for {name} ...")
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=grid,
            n_iter=n_iter,
            scoring="accuracy",
            cv=3,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        search.fit(X_tr[FEATURE_COLUMNS], y_tr)
        best_model = search.best_estimator_
        
        # Evaluate on test set
        proba_all = best_model.predict_proba(X_te[FEATURE_COLUMNS])
        if proba_all.shape[1] == 1:
            preds = np.ones(len(X_te)) if best_model.classes_[0] == 1 else np.zeros(len(X_te))
            auc = 0.5
        else:
            preds = (proba_all[:, 1] >= 0.5).astype(int)
            try:
                auc = roc_auc_score(y_te, proba_all[:, 1])
            except ValueError:
                auc = 0.5
                
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, zero_division=0)
        
        results[name] = {
            "model": best_model,
            "params": search.best_params_,
            "val_acc": search.best_score_,
            "test_acc": acc,
            "test_f1": f1,
            "test_auc": auc
        }
        
    # Print comparison scorecard
    print("\n" + "=" * 90)
    print("  PHYSICAL RISK CLASSIFICATION MODEL COMPARISON (Spoilage Prediction)")
    print("=" * 90)
    print(f"  {'Algorithm':<30} | {'CV Val Acc':<12} | {'Test Acc':<10} | {'Test F1':<8} | {'Test AUC':<8}")
    print("-" * 90)
    for name, res in results.items():
        print(f"  {name:<30} | {res['val_acc']:<12.4f} | {res['test_acc']:<10.4f} | {res['test_f1']:<8.4f} | {res['test_auc']:<8.4f}")
    print("=" * 90)
    
    # Choose best algorithm based on highest test accuracy
    best_name = max(results, key=lambda k: results[k]["test_acc"])
    logger.info(f"Selected Best Physical Model: {best_name} (Test Accuracy: {results[best_name]['test_acc']*100:.2f}%)")
    
    return results[best_name]["model"], best_name, results[best_name]["params"]

def save_model(model, directory, model_type, commodity_name):
    os.makedirs(directory, exist_ok=True)
    filename = f"{model_type.lower()}_{commodity_name.lower()}.pkl"
    path = os.path.join(directory, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved Tuned Model → {path}")
    return path

def run_tuning(target_commodity, n_iter, save):
    # Select commodities to tune
    selected = [target_commodity] if target_commodity in COMMODITIES else COMMODITIES
    
    logger.info("=" * 80)
    print("       CropCredit ML Hyperparameter Tuning & Multi-Algorithm Comparison")
    logger.info("=" * 80)
    logger.info(f"Target Commodities : {selected}")
    logger.info(f"Search Iterations  : {n_iter}")
    logger.info(f"Auto-Save Models   : {save}")
    logger.info("=" * 80)

    for commodity in selected:
        logger.info(f"\n[+] Tuning & comparing algorithms for crop: {commodity}")
        df = load_real_data(commodity)
        if df is None:
            logger.error(f"Real dataset for {commodity} not found. Skipping.")
            continue
            
        X = df[FEATURE_COLUMNS]
        
        # 1. Financial Regressors Tuning & Comparison
        y_fin = df["future_price"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_fin, test_size=0.20, random_state=RANDOM_STATE)
        
        best_fin, name_fin, params_fin = tune_regressors(X_tr, y_tr, X_te, y_te, n_iter=n_iter)
        
        # 2. Physical Classifiers Tuning & Comparison
        y_phy = df["spoilage_label"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_phy, test_size=0.20, random_state=RANDOM_STATE)
        
        best_phy, name_phy, params_phy = tune_classifiers(X_tr, y_tr, X_te, y_te, n_iter=n_iter)
        
        # Save best models
        if save:
            logger.info(f"\n[!] Saving best models for {commodity}...")
            save_model(best_fin, MODELS_DIR, "financialriskmodel", commodity)
            save_model(best_phy, MODELS_DIR, "physicalriskmodel", commodity)

    logger.info("\n" + "=" * 80)
    logger.info("  Multi-Algorithm Tuning Pipeline Complete.")
    logger.info("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CropCredit - Multi-algorithm hyperparameter tuning utility"
    )
    parser.add_argument(
        "--commodity", type=str, default="all",
        help="Specific crop to tune (Wheat, Rice, Potato, Onion, Tomato, or all)"
    )
    parser.add_argument(
        "--n-iter", type=int, default=10,
        help="Number of iterations for RandomizedSearchCV (default: 10)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the best tuned models directly, replacing existing ones"
    )
    args = parser.parse_args()
    
    commodity_input = args.commodity.strip().capitalize()
    
    run_tuning(commodity_input, args.n_iter, args.save)
