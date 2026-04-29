"""
predict.py
- Load a trained model
- Predict delay days for a single input row
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import joblib
import pandas as pd

from modeling import add_time_features, add_derived_features


def load_model(model_path: str):
    return joblib.load(model_path)


def validate_payload(payload: dict) -> None:
    required = {
        "planned_shipment_date",
        "team_size",
        "feature_complexity",
        "num_dependencies",
        "sprint_length_weeks",
        "num_blockers",
        "holidays_in_sprint",
        "priority_encoded",
        "past_avg_delay_days",
        "estimated_bug_count",
    }
    missing = required.difference(payload.keys())
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise SystemExit(f"Missing required fields: {missing_list}")


def prepare_single_row(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame([payload])
    df = add_time_features(df, "planned_shipment_date")
    df = add_derived_features(df)
    df = df.drop(columns=["planned_shipment_date"])
    return df


def reorder_features(df: pd.DataFrame, feature_order: list) -> pd.DataFrame:
    """
    Reorder features to match the order used during model training.

    Args:
        df (pd.DataFrame): Input dataframe.
        feature_order (list): List of feature names in the correct order.

    Returns:
        pd.DataFrame: Reordered dataframe.
    """
    return df[feature_order]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict delay_days for a single record")
    project_root = Path(__file__).resolve().parents[1]
    default_model = project_root / "models" / "baseline_rf.joblib"

    parser.add_argument("--model", default=str(default_model), help="Path to trained model")
    parser.add_argument("--json", help="JSON string for one input row")
    parser.add_argument("--json-file", help="Path to JSON file for one input row")
    args = parser.parse_args()

    model = load_model(args.model)
    if args.json_file:
        with open(args.json_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    elif args.json:
        payload = json.loads(args.json)
    else:
        raise SystemExit("Provide --json or --json-file")

    validate_payload(payload)
    payload["planned_shipment_date"] = pd.to_datetime(payload["planned_shipment_date"], errors="coerce")
    if pd.isna(payload["planned_shipment_date"]):
        raise SystemExit("Invalid planned_shipment_date. Use YYYY-MM-DD format.")
    X = prepare_single_row(payload)

    # Ensure feature order matches training
    feature_order = model.feature_names_in_
    X = reorder_features(X, feature_order)

    delay_days = round(model.predict(X)[0])  # Round to nearest integer
    predicted_shipment_date = payload["planned_shipment_date"] + pd.to_timedelta(delay_days, unit="D")

    print("Predicted delay_days:", delay_days)
    print("Predicted shipment date:", predicted_shipment_date.strftime("%Y-%m-%d"))


if __name__ == "__main__":
    main()
