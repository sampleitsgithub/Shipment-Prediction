"""
tuning.py
- Hyperparameter tuning with Optuna (RandomForest baseline)
- Optimize MAE on a validation split
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import joblib
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from modeling import add_time_features, add_derived_features
from utils.data_utils import apply_outlier_bounds, compute_outlier_bounds, time_based_split


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def prepare_features(df: pd.DataFrame):
    df = add_time_features(df, "planned_shipment_date")
    df = add_derived_features(df)
    y = df["delay_days"]
    X = df.drop(columns=["delay_days", "planned_shipment_date"])
    return X, y


def objective(trial, X_train, X_valid, y_train, y_valid):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": 42,
        "n_jobs": -1,
    }

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds)
    return mae


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    project_root = Path(__file__).resolve().parents[1]
    default_data = project_root / "data" / "dataset.csv"
    default_out = project_root / "models" / "tuned_rf.joblib"

    parser.add_argument("--data", default=str(default_data), help="Path to CSV")
    parser.add_argument("--out", default=str(default_out), help="Output model path")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()

    df = load_data(args.data)
    train_df, valid_df = time_based_split(df, "planned_shipment_date")

    # Cap outliers using bounds from training only
    bounds = compute_outlier_bounds(train_df, "delay_days")
    train_df = apply_outlier_bounds(train_df, "delay_days", bounds)
    valid_df = apply_outlier_bounds(valid_df, "delay_days", bounds)

    X_train, y_train = prepare_features(train_df)
    X_valid, y_valid = prepare_features(valid_df)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, X_train, X_valid, y_train, y_valid), n_trials=args.trials)

    best_params = study.best_params
    best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, args.out)

    print("Best MAE:", study.best_value)
    print("Best params:", best_params)
    print(f"Model saved to: {args.out}")


if __name__ == "__main__":
    main()
