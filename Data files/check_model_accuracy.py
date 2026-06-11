
import os
import sys
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
MODELS_DIR = "models"

COMMODITIES = ["Wheat", "Rice", "Potato", "Onion", "Tomato"]

FEATURE_COLUMNS = [
    "commodity_encoded",
    "tonnage",
    "market_arrivals",
    "current_price",
    "rainfall_deficit",
    "warehouse_temp",
    "humidity",
    "moisture_content",
]
TARGET_PRICE   = "future_price"
TARGET_SPOILAGE= "spoilage_label"

# ── Colour helpers for terminal ────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def score_badge(value: float, thresholds: tuple) -> str:
    """Return coloured badge based on thresholds (good, ok)."""
    good, ok = thresholds
    if value >= good:
        return f"{GREEN} Excellent{RESET}"
    elif value >= ok:
        return f"{YELLOW} Good{RESET}"
    else:
        return f"{RED} Needs improvement{RESET}"

def sep(char="-", n=66):
    print(char * n)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════
def load_models(commodity: str):
    """Load the commodity-specific models and the label encoder."""
    paths = {
        "financial": os.path.join(MODELS_DIR, f"financialriskmodel_{commodity.lower()}.pkl"),
        "physical":  os.path.join(MODELS_DIR, f"physicalriskmodel_{commodity.lower()}.pkl"),
        "encoder":   os.path.join(MODELS_DIR, "label_encoder.pkl"),
    }
    models = {}
    for key, path in paths.items():
        if not os.path.exists(path):
            print(f"{RED}  Not found: {path}{RESET}")
            print(f"   Run python train_models.py first.\n")
            sys.exit(1)
        with open(path, "rb") as f:
            models[key] = pickle.load(f)
        print(f"    Loaded: {os.path.basename(path)}")
    return models["financial"], models["physical"], models["encoder"]


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════
def load_csv(csv_path: str) -> pd.DataFrame:
    """Load and validate a CSV for scoring."""
    if not os.path.exists(csv_path):
        print(f"{RED}  CSV not found: {csv_path}{RESET}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"    Loaded: {os.path.basename(csv_path)}  ({len(df):,} rows)")

    required = FEATURE_COLUMNS + [TARGET_PRICE, TARGET_SPOILAGE]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"{RED}  Missing columns in CSV: {missing}{RESET}")
        sys.exit(1)

    # Clean
    df = df.dropna(subset=FEATURE_COLUMNS)
    df[TARGET_SPOILAGE] = df[TARGET_SPOILAGE].clip(0, 1).astype(int)
    df = df[df[TARGET_PRICE] > 0]
    print(f"    After cleaning: {len(df):,} valid rows")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ACCURACY CHECKER
# ══════════════════════════════════════════════════════════════════════════════
def check_financial_model(model, df: pd.DataFrame):
    """Score the XGBoost regression model on the CSV data."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_PRICE]

    preds = model.predict(X)

    mae  = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2   = r2_score(y, preds)
    mape = mean_absolute_percentage_error(y, preds) * 100

    print()
    sep("=")
    print(f"{BOLD}{CYAN}  PILLAR 1 - FINANCIAL RISK MODEL (Price Oracle){RESET}")
    print(f"{CYAN}  XGBoost Regressor | Target: future_price{RESET}")
    sep("=")
    print(f"  Rows evaluated    : {len(df):,}")
    print()
    print(f"  Accuracy Metrics ")
    print(f"  R^2 Score          : {BOLD}{r2:.4f}{RESET}  {score_badge(r2, (0.90, 0.75))}")
    print(f"  MAE  (Rs./quintal) : {BOLD}Rs.{mae:>10,.2f}{RESET}  (lower is better)")
    print(f"  RMSE (Rs./quintal) : {BOLD}Rs.{rmse:>10,.2f}{RESET}  (lower is better)")
    print(f"  MAPE              : {BOLD}{mape:>8.2f}%{RESET}  "
          f"{score_badge(100 - mape, (90, 80))} (100-MAPE)")
    print()

    # Show sample predictions vs actuals
    print(f"  -- Sample Predictions vs Actuals (first 10 rows) ---------")
    sample = pd.DataFrame({
        "Actual Price (Rs.)":    y.values[:10],
        "Predicted Price (Rs.)": preds[:10],
        "Error (Rs.)":           preds[:10] - y.values[:10],
        "Error %":             (preds[:10] - y.values[:10]) / y.values[:10] * 100,
    }).round(2)
    print(sample.to_string(index=False))
    sep()

    # Price range accuracy
    for pct in [5, 10, 20]:
        within = np.mean(np.abs(preds - y.values) / y.values * 100 <= pct) * 100
        status = f"{GREEN}[PASS]{RESET}" if within > 80 else f"{YELLOW}[OK]{RESET}" if within > 60 else f"{RED}[FAIL]{RESET}"
        print(f"  Within +/-{pct:2d}% of actual : {within:.1f}%  {status}")

    sep()
    return {"r2": r2, "mae": mae, "rmse": rmse, "mape": mape}


def check_physical_model(model, df: pd.DataFrame, threshold: float = 0.50):
    """Score the Random Forest classification model on the CSV data."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_SPOILAGE]

    proba_all = model.predict_proba(X)
    if proba_all.shape[1] == 1:
        proba = np.ones(len(X)) if model.classes_[0] == 1 else np.zeros(len(X))
    else:
        proba = proba_all[:, 1]
        
    preds = (proba >= threshold).astype(int)

    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds,    zero_division=0)
    f1   = f1_score(y, preds,        zero_division=0)
    try:
        auc  = roc_auc_score(y, proba)
    except ValueError:
        auc  = 0.0
    cm   = confusion_matrix(y, preds, labels=[0, 1])

    print()
    sep("=")
    print(f"{BOLD}{CYAN}  PILLAR 2 - PHYSICAL RISK MODEL (Spoilage Classifier){RESET}")
    print(f"{CYAN}  Random Forest | Target: spoilage_label  "
          f"| Threshold: {threshold}{RESET}")
    sep("=")
    print(f"  Rows evaluated    : {len(df):,}")
    class_dist = y.value_counts()
    print(f"  Class 0 (Low)     : {class_dist.get(0,0):,}  ({class_dist.get(0,0)/len(df):.1%})")
    print(f"  Class 1 (High)    : {class_dist.get(1,0):,}  ({class_dist.get(1,0)/len(df):.1%})")
    print()
    print(f"  Accuracy Metrics ")
    print(f"  Accuracy          : {BOLD}{acc:.4f}  ({acc*100:.2f}%){RESET}  {score_badge(acc, (0.90, 0.80))}")
    print(f"  Precision         : {BOLD}{prec:.4f}{RESET}  (of flagged HIGH, this % is correct)")
    print(f"  Recall            : {BOLD}{rec:.4f}{RESET}  (of actual HIGH, this % caught)")
    print(f"  F1 Score          : {BOLD}{f1:.4f}{RESET}  {score_badge(f1, (0.90, 0.80))}")
    print(f"  AUC-ROC           : {BOLD}{auc:.4f}{RESET}  {score_badge(auc, (0.95, 0.85))}")
    print()

    print(f"  Confusion Matrix ")
    tn, fp_v, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total = tn + fp_v + fn + tp
    print(f"                      Pred LOW     Pred HIGH")
    print(f"  Actual LOW    :    {tn:>8,}    {fp_v:>8,}")
    print(f"  Actual HIGH   :    {fn:>8,}    {tp:>8,}")
    print()
    print(f"  True  Negatives : {tn:,}   ({tn/total:.1%})   Correct LOW")
    print(f"  False Positives : {fp_v:,}   ({fp_v/total:.1%})   False alarm")
    miss_color = GREEN if fn/total < 0.05 else YELLOW if fn/total < 0.15 else RED
    print(f"  False Negatives : {miss_color}{fn:,}   ({fn/total:.1%})   Missed spoilage (critical!){RESET}")
    print(f"  True  Positives : {tp:,}   ({tp/total:.1%})   Correct HIGH")
    print()

    print(f"  Full Classification Report ")
    report = classification_report(y, preds, labels=[0, 1], target_names=["Low Risk","High Risk"], zero_division=0)
    for line in report.split("\n"):
        print(f"  {line}")

    print()
    print(f"  Probability Distribution ")
    bins  = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]
    hist, _ = np.histogram(proba, bins=bins)
    for i, count in enumerate(hist):
        bar = "#" * min(40, int(count / len(proba) * 200))
        pct = count / len(proba) * 100
        print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}] : {bar}  {count:,} ({pct:.1f}%)")

    sep()
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc_roc": auc}


def print_final_verdict(fin: dict, phy: dict):
    print()
    sep("=")
    print(f"{BOLD}  FINAL SCORECARD{RESET}")
    sep("=")
    print(f"  {'Model':<35} {'Score':<12} {'Status'}")
    sep()
    print(f"  {'Pillar 1 - R^2 (Price Accuracy)':<35} {fin['r2']:<12.4f} "
          f"{score_badge(fin['r2'], (0.90, 0.75))}")
    print(f"  {'Pillar 1 - MAE (Rs./quintal)':<35} Rs.{fin['mae']:<10,.0f} "
          f"(lower is better)")
    print(f"  {'Pillar 2 - Accuracy':<35} {phy['accuracy']*100:<10.2f}% "
          f"{score_badge(phy['accuracy'], (0.90, 0.80))}")
    print(f"  {'Pillar 2 - AUC-ROC':<35} {phy['auc_roc']:<12.4f} "
          f"{score_badge(phy['auc_roc'], (0.95, 0.85))}")
    print(f"  {'Pillar 2 - F1 Score':<35} {phy['f1']:<12.4f} "
          f"{score_badge(phy['f1'], (0.90, 0.80))}")
    sep()

    p1_ready = fin["r2"] > 0.75
    p2_ready = phy["accuracy"] > 0.80

    if p1_ready and p2_ready:
        print(f"  {GREEN}{BOLD}  VERDICT: Both models are production-ready.{RESET}")
        print(f"   Run the app: streamlit run app.py")
    elif p1_ready or p2_ready:
        weak = "Pillar 2" if p1_ready else "Pillar 1"
        print(f"  {YELLOW}  VERDICT: {weak} needs more data or tuning.{RESET}")
    else:
        print(f"  {RED}  VERDICT: Both models need improvement.{RESET}")
        
    sep("=")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="CropCredit — Check trained model accuracy on CSV data"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to a CSV file to score (default: auto-finds cleaned_*.csv)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.50,
        help="Probability threshold for spoilage HIGH classification (default: 0.50)"
    )
    args = parser.parse_args()

    print()
    sep("=")
    print(f"  {BOLD}CropCredit - Model Accuracy Checker{RESET}")
    print(f"  Models dir: ./{MODELS_DIR}/")
    sep("=")

    # ── Find CSV ─────────────────────────────────────────────────────────────
    if args.csv:
        csv_path = args.csv
    else:
        import glob
        candidates = (glob.glob("cleaned_*.csv") + glob.glob("*.csv") +
                      glob.glob("Data files/cleaned_*.csv") + glob.glob("Data files/*.csv"))
        # Exclude weather/meta files
        candidates = [c for c in candidates
                      if "weather" not in c.lower() and "cleaning" not in c.lower()]
        if not candidates:
            print(f"{RED}  No CSV files found. Pass --csv <file.csv>{RESET}")
            sys.exit(1)
        # Pick largest file (most data)
        csv_path = max(candidates, key=os.path.getsize)
        print(f"\n  Auto-selected: {csv_path}")
        print(f"  (Use --csv <file> to score a specific file)")

    # Deduce commodity from CSV filename
    basename = os.path.basename(csv_path).lower()
    commodity = None
    for c in COMMODITIES:
        if c.lower() in basename:
            commodity = c
            break
    if not commodity:
        print(f"{YELLOW}  Could not deduce commodity from file name: {csv_path}. Defaulting to Wheat.{RESET}")
        commodity = "Wheat"

    # ── Load models ──────────────────────────────────────────────────────────
    print(f"\n  Loading models for {commodity} ...")
    fin_model, phy_model, _ = load_models(commodity)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n  Loading data ...")
    df = load_csv(csv_path)

    # ── Score both models ─────────────────────────────────────────────────────
    fin_metrics = check_financial_model(fin_model, df)
    phy_metrics = check_physical_model(phy_model, df, threshold=args.threshold)

    # ── Final verdict ─────────────────────────────────────────────────────────
    print_final_verdict(fin_metrics, phy_metrics)


if __name__ == "__main__":
    main()
