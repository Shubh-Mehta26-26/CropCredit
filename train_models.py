import os
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_absolute_error, r2_score,
                             classification_report, roc_auc_score)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CropCredit.Trainer")

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR   = "models"
RANDOM_STATE = 42

# Matches COMMODITIES in app.py exactly — encoder must be consistent
COMMODITIES = [
    "Wheat","Rice","Potato","Onion","Tomato",
]

BASE_PRICES = {
    "Wheat":2275,"Rice":2183,"Potato":1200,"Onion":1800,"Tomato":2200,
}

SPOILAGE_SENSITIVITY = {
    "Wheat":0.7,"Rice":0.9,"Potato":1.3,"Onion":1.0,"Tomato":2.0,
}

# Feature columns — must match app.py exactly
FEATURE_COLUMNS = [
    "commodity_encoded","tonnage","market_arrivals",
    "current_price","rainfall_deficit",
    "warehouse_temp","humidity","moisture_content",
]

# ════════════════════════════════════════════════════════════════════════════
#  PROCEDURAL RISK MODEL FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════
def check_features(df):
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

def get_financial_model():
    return XGBRegressor(
        n_estimators=700, learning_rate=0.04, max_depth=6,
        subsample=0.80, colsample_bytree=0.80,
        reg_alpha=0.10, reg_lambda=1.00, min_child_weight=5,
        random_state=RANDOM_STATE, n_jobs=-1,
        tree_method="hist", eval_metric="mae",
    )

def get_physical_model():
    return RandomForestClassifier(
        n_estimators=500, max_depth=12,
        min_samples_split=8, min_samples_leaf=4,
        max_features="sqrt", class_weight="balanced",
        oob_score=True, random_state=RANDOM_STATE, n_jobs=-1,
    )

def train_model(model, X, y):
    logger.info(f"Training on {X.shape[0]:,} samples ...")
    model.fit(X[FEATURE_COLUMNS], y)
    logger.info("Training complete.")

def predict_financial(model, X):
    check_features(X)
    return model.predict(X[FEATURE_COLUMNS])

def predict_physical(model, X):
    check_features(X)
    proba_all = model.predict_proba(X[FEATURE_COLUMNS])
    if proba_all.shape[1] == 1:
        if model.classes_[0] == 1:
            return np.ones(len(X))
        else:
            return np.zeros(len(X))
    return proba_all[:, 1]

def evaluate_financial(model, X, y):
    preds = predict_financial(model, X)
    mae   = mean_absolute_error(y, preds)
    r2    = r2_score(y, preds)
    cv    = cross_val_score(model, X[FEATURE_COLUMNS], y,
                            cv=5, scoring="r2", n_jobs=-1)
    metrics = {
        "MAE (INR/quintal)": round(mae, 2),
        "R² Score":          round(r2,  4),
        "CV R² Mean":        round(cv.mean(), 4),
        "CV R² Std":         round(cv.std(),  4),
    }
    logger.info(f"Metrics | {metrics}")
    return metrics

def evaluate_physical(model, X, y):
    proba = predict_physical(model, X)
    preds = (proba >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y, proba)
    except ValueError:
        auc = 0.5  # ROC AUC is undefined when only one class is present in y
    oob = getattr(model, "oob_score_", None)
    
    # Determine labels dynamically based on present classes
    unique_y = np.unique(y)
    target_names = []
    if 0 in unique_y:
        target_names.append("Low Risk")
    if 1 in unique_y:
        target_names.append("High Risk")
    if not target_names:
        target_names = ["Low Risk", "High Risk"]
        
    logger.info("\n" + classification_report(y, preds,
                  target_names=target_names, labels=unique_y, zero_division=0))
    metrics = {
        "AUC-ROC":      round(auc, 4),
        "OOB Accuracy": round(oob, 4) if oob else "N/A",
    }
    logger.info(f"Metrics | {metrics}")
    return metrics

def save_model(model, directory, model_type, commodity_name=None):
    os.makedirs(directory, exist_ok=True)
    suffix = f"_{commodity_name.lower()}" if commodity_name else ""
    filename = f"{model_type.lower()}{suffix}.pkl"
    path = os.path.join(directory, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved → {path}")
    return path

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

def save_encoder(le):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    with open(path, "wb") as f:
        pickle.dump(le, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"LabelEncoder saved → {path}")
    logger.info(f"Classes: {list(le.classes_)}")

def run_pipeline():
    logger.info("=" * 62)
    logger.info("  CropCredit ML Training Pipeline — START")
    logger.info("=" * 62)

    le = LabelEncoder().fit(COMMODITIES)
    save_encoder(le)

    for commodity in COMMODITIES:
        logger.info(f"\nProcessing commodity: {commodity}")
        df = load_real_data(commodity)
        if df is None:
            raise FileNotFoundError(f"Real data file for {commodity} not found!")

        X = df[FEATURE_COLUMNS]
        
        # Financial Risk Model
        logger.info(f"  Model  : FinancialRiskModel ({commodity})")
        fin_model = get_financial_model()
        y_fin = df["future_price"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_fin, test_size=0.20, random_state=RANDOM_STATE)
        train_model(fin_model, X_tr, y_tr)
        evaluate_financial(fin_model, X_te, y_te)
        save_model(fin_model, MODELS_DIR, "financialriskmodel", commodity)

        # Physical Risk Model
        logger.info(f"  Model  : PhysicalRiskModel ({commodity})")
        phy_model = get_physical_model()
        y_phy = df["spoilage_label"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_phy, test_size=0.20, random_state=RANDOM_STATE)
        train_model(phy_model, X_tr, y_tr)
        evaluate_physical(phy_model, X_te, y_te)
        save_model(phy_model, MODELS_DIR, "physicalriskmodel", commodity)

    logger.info("\n" + "=" * 62)
    logger.info("  Pipeline COMPLETE — models saved to ./" + MODELS_DIR)
    logger.info("=" * 62)

if __name__ == "__main__":
    run_pipeline()
