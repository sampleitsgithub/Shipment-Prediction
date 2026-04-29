# utils/data_utils.py
# Utility functions for data handling

import pandas as pd


def compute_outlier_bounds(df: pd.DataFrame, column: str) -> tuple[float, float]:
    """Compute IQR bounds for outlier capping from training data only."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    iqr = Q3 - Q1
    lower_bound = Q1 - 1.5 * iqr
    upper_bound = Q3 + 1.5 * iqr
    return lower_bound, upper_bound


def apply_outlier_bounds(df: pd.DataFrame, column: str, bounds: tuple[float, float]) -> pd.DataFrame:
    """Apply precomputed bounds to cap outliers without leakage."""
    lower_bound, upper_bound = bounds
    df = df.copy()
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

def time_based_split(df: pd.DataFrame, date_col: str, split_ratio: float = 0.8):
    """
    Perform a time-based split on the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        date_col (str): Column name for datetime sorting.
        split_ratio (float): Ratio for training data.

    Returns:
        pd.DataFrame, pd.DataFrame: Training and validation dataframes.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(by=date_col)
    split_index = int(len(df) * split_ratio)
    train_df = df.iloc[:split_index]
    valid_df = df.iloc[split_index:]
    return train_df, valid_df