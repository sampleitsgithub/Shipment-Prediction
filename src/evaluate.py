"""
evaluate.py
- Load trained model
- Run evaluation on a holdout split
- Report MAE and basic diagnostics
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from modeling import add_time_features, add_derived_features
from utils.data_utils import apply_outlier_bounds, compute_outlier_bounds, time_based_split


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = add_time_features(df, "planned_shipment_date")
    df = add_derived_features(df)
    y = df["delay_days"]
    X = df.drop(columns=["delay_days", "planned_shipment_date"])
    return X, y


def evaluate_model(model_path: Path, X_valid: pd.DataFrame, y_valid: pd.Series):
    model = joblib.load(model_path)
    preds = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds)
    r2 = r2_score(y_valid, preds)
    return model, mae, r2


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    project_root = Path(__file__).resolve().parents[1]
    default_data = project_root / "data" / "dataset.csv"
    default_baseline = project_root / "models" / "baseline_rf.joblib"
    default_tuned = project_root / "models" / "tuned_rf.joblib"

    parser.add_argument("--data", default=str(default_data), help="Path to CSV")
    parser.add_argument("--baseline-model", default=str(default_baseline), help="Path to baseline model")
    parser.add_argument("--tuned-model", default=str(default_tuned), help="Path to tuned model")
    args = parser.parse_args()

    df = load_data(args.data)
    train_df, valid_df = time_based_split(df, "planned_shipment_date")

    # Cap outliers using bounds from training only
    bounds = compute_outlier_bounds(train_df, "delay_days")
    valid_df = apply_outlier_bounds(valid_df, "delay_days", bounds)

    X_valid, y_valid = prepare_features(valid_df)

    baseline_path = Path(args.baseline_model)
    tuned_path = Path(args.tuned_model)

    print("\nModel Evaluation (time-based split)")
    print("Model\t\tMAE\tR2")
    if baseline_path.exists():
        _, baseline_mae, baseline_r2 = evaluate_model(baseline_path, X_valid, y_valid)
        print(f"Baseline RF\t{baseline_mae:.3f}\t{baseline_r2:.3f}")
    else:
        print("Baseline RF\tN/A\tN/A")

    if tuned_path.exists():
        tuned_model, tuned_mae, tuned_r2 = evaluate_model(tuned_path, X_valid, y_valid)
        print(f"Tuned RF\t{tuned_mae:.3f}\t{tuned_r2:.3f}")
        importances = pd.Series(tuned_model.feature_importances_, index=X_valid.columns)
        print("\nTop 10 Feature Importances (Tuned RF)")
        print(importances.sort_values(ascending=False).head(10).to_string())
    else:
        print("Tuned RF\tN/A\tN/A")


if __name__ == "__main__":
    main()
