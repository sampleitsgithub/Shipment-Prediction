# Predictive Modeling for Feature Shipment Prediction

## Problem Statement
Build a machine learning model that predicts `delay_days` for software feature shipments using historical data. The solution must be time-series aware, minimize MAE, and be packaged cleanly.

## Dataset
Input file: `data/dataset.csv`

Key columns include:
- `planned_shipment_date`
- `delay_days` (target)
- operational and engineering signals (team size, blockers, complexity, etc.)

## Approach
1) **EDA & Preprocessing**
	- Inspect missing values and basic statistics
	- Identify correlations and potential outliers

2) **Feature Engineering**
	- Time-based features from `planned_shipment_date`
	- Interaction and ratio features

3) **Model Selection & Baseline**
	- RandomForestRegressor baseline
	- Time-based split (first 80% train, last 20% validation)

4) **Fine-Tuning**
	- Optuna to minimize MAE on the validation split

## Results
Run evaluation to compute MAE:
- `python src/evaluate.py`

| Model       | MAE  | R2   |
|-------------|------|------|
| Baseline RF | 0.555 | 0.978 |
| Tuned RF    | 0.616 | 0.974 |

## How to Run
From the project root (with `venv` activated):

1) EDA and preprocessing
	- `python src/data_preprocessing.py`

2) Train baseline model
	- `python src/modeling.py`

3) Evaluate model
	- `python src/evaluate.py`

4) Hyperparameter tuning
	- `python src/tuning.py --trials 50`

5) Single prediction
	- `python src/predict.py --json-file reports/prediction_input.json`

## Project Structure
- `data/` : Dataset
- `src/` : Core scripts (EDA, modeling, tuning, prediction)
- `models/` : Saved models
- `reports/` : Generated artifacts
- `utils/` : Shared utilities

