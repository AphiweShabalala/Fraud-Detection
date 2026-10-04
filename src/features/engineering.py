import numpy as np
import pandas as pd


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cyclical time features from the Time column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the Time column.

    Returns
    -------
    pd.DataFrame
        DataFrame with Hour_sin and Hour_cos added.
    """
    if "Time" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Time' column.")
    df = df.copy()

    df["Hour"] = (df["Time"] // 3600) % 24

    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    df = df.drop(columns=["Hour"])

    return df