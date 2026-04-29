"""
modeling.py
- Feature engineering
- Train baseline model
- Evaluate with MAE
- Save trained model
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from utils.data_utils import apply_outlier_bounds, compute_outlier_bounds, time_based_split


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # Drop rows with invalid dates to keep time-based split reliable
    df = df.dropna(subset=[date_col])
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["weekofyear"] = df[date_col].dt.isocalendar().week.astype(int)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["deps_per_team"] = df["num_dependencies"] / df["team_size"].replace(0, np.nan)
    df["blockers_per_week"] = df["num_blockers"] / df["sprint_length_weeks"].replace(0, np.nan)
    df["bugs_per_complexity"] = df["estimated_bug_count"] / df["feature_complexity"].replace(0, np.nan)
    df["deps_blockers"] = df["num_dependencies"] * df["num_blockers"]
    df["complexity_team"] = df["feature_complexity"] * df["team_size"]
    df = df.fillna(0)
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = add_time_features(df, "planned_shipment_date")
    df = add_derived_features(df)

    # Target: delay_days (regression)
    y = df["delay_days"]

    # Remove planned_shipment_date before modeling
    X = df.drop(columns=["delay_days", "planned_shipment_date"]).copy()

    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate RandomForest model")
    project_root = Path(__file__).resolve().parents[1]
    default_data = project_root / "data" / "dataset.csv"
    default_model = project_root / "models" / "baseline_rf.joblib"

    parser.add_argument("--data", default=str(default_data), help="Path to CSV")
    parser.add_argument("--out", default=str(default_model), help="Output model path")
    args = parser.parse_args()

    df = load_data(args.data)
    train_df, valid_df = time_based_split(df, "planned_shipment_date")

    # Cap outliers using bounds from training only
    bounds = compute_outlier_bounds(train_df, "delay_days")
    train_df = apply_outlier_bounds(train_df, "delay_days", bounds)
    valid_df = apply_outlier_bounds(valid_df, "delay_days", bounds)

    X_train, y_train = prepare_features(train_df)
    X_valid, y_valid = prepare_features(valid_df)

    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)

    print("Validation MAE:", mae)
    print(f"Model saved to: {args.out}")


if __name__ == "__main__":
    main()
