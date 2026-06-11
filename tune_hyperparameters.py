import os
import sys
import pickle
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score
from xgboost import XGBRegressor

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

def tune_financial_model(X, y, n_iter=10):
    logger.info(f"Tuning FinancialRiskModel (XGBoost Regressor) with {n_iter} iterations ...")
    
    base_model = XGBRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mae"
    )
    
    param_dist = {
        "n_estimators": [100, 200, 400, 600, 800],
        "learning_rate": [0.01, 0.02, 0.04, 0.08, 0.1, 0.15],
        "max_depth": [4, 5, 6, 7, 8],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
        "min_child_weight": [1, 3, 5, 7]
    }
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    search.fit(X[FEATURE_COLUMNS], y)
    logger.info(f"Financial Model Tuning Complete. Best MAE: {-search.best_score_:.2f}")
    return search.best_estimator_, search.best_params_, -search.best_score_

def tune_physical_model(X, y, n_iter=10):
    logger.info(f"Tuning PhysicalRiskModel (Random Forest Classifier) with {n_iter} iterations ...")
    
    base_model = RandomForestClassifier(
        class_weight="balanced",
        oob_score=True,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    param_dist = {
        "n_estimators": [100, 200, 300, 500, 700],
        "max_depth": [6, 8, 10, 12, 14, 16, None],
        "min_samples_split": [2, 4, 6, 8, 10],
        "min_samples_leaf": [1, 2, 4, 6],
        "max_features": ["sqrt", "log2", None]
    }
    
    # Check if we only have 1 class in y. If so, CV will fail.
    if len(np.unique(y)) < 2:
        logger.warning("Only one target class present. Skipping CV search and using default model.")
        base_model.fit(X[FEATURE_COLUMNS], y)
        return base_model, {}, 1.0
        
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="accuracy",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    search.fit(X[FEATURE_COLUMNS], y)
    logger.info(f"Physical Model Tuning Complete. Best Accuracy: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.best_score_

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
    
    logger.info("=" * 66)
    logger.info("  CropCredit ML Hyperparameter Tuning Pipeline")
    logger.info("=" * 66)
    logger.info(f"Target Commodities : {selected}")
    logger.info(f"Search Iterations  : {n_iter}")
    logger.info(f"Auto-Save Models   : {save}")
    logger.info("=" * 66)

    for commodity in selected:
        logger.info(f"\n[+] Tuning parameters for crop: {commodity}")
        df = load_real_data(commodity)
        if df is None:
            logger.error(f"Real dataset for {commodity} not found. Skipping.")
            continue
            
        X = df[FEATURE_COLUMNS]
        
        # 1. Financial Regressor Tuning
        y_fin = df["future_price"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_fin, test_size=0.20, random_state=RANDOM_STATE)
        
        best_fin, params_fin, score_fin = tune_financial_model(X_tr, y_tr, n_iter=n_iter)
        
        # Evaluate on test set
        preds_te = best_fin.predict(X_te[FEATURE_COLUMNS])
        mae_te = mean_absolute_error(y_te, preds_te)
        r2_te = r2_score(y_te, preds_te)
        logger.info(f"Tuned Financial Model Test MAE: Rs.{mae_te:.2f} | R^2: {r2_te:.4f}")
        logger.info(f"Best Hyperparameters: {params_fin}")
        
        # 2. Physical Classifier Tuning
        y_phy = df["spoilage_label"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_phy, test_size=0.20, random_state=RANDOM_STATE)
        
        best_phy, params_phy, score_phy = tune_physical_model(X_tr, y_tr, n_iter=n_iter)
        
        # Evaluate on test set
        proba_te = best_phy.predict_proba(X_te[FEATURE_COLUMNS])
        if proba_te.shape[1] == 1:
            preds_phy = np.ones(len(X_te)) if best_phy.classes_[0] == 1 else np.zeros(len(X_te))
        else:
            preds_phy = (proba_te[:, 1] >= 0.5).astype(int)
            
        acc_te = accuracy_score(y_te, preds_phy)
        logger.info(f"Tuned Physical Model Test Accuracy: {acc_te*100:.2f}%")
        logger.info(f"Best Hyperparameters: {params_phy}")
        
        # Save
        if save:
            save_model(best_fin, MODELS_DIR, "financialriskmodel", commodity)
            save_model(best_phy, MODELS_DIR, "physicalriskmodel", commodity)

    logger.info("\n" + "=" * 66)
    logger.info("  Tuning Pipeline Complete.")
    logger.info("=" * 66)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CropCredit - Hyperparameter tuning utility"
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
