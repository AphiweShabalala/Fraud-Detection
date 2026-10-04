import pandas as pd


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str]
) -> None:
    """
    Validate that all required columns are present.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    required_columns : list[str]
        Columns that must exist in the dataframe.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def validate_no_missing_values(df: pd.DataFrame) -> None:
    """
    Validate that the dataframe contains no missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Raises
    ------
    ValueError
        If one or more columns contain missing values.
    """

    missing_counts = df.isnull().sum()

    columns_with_missing = missing_counts[
        missing_counts > 0
    ]

    if not columns_with_missing.empty:
        raise ValueError(
            f"Missing values found:\n{columns_with_missing}"
        )

def validate_target(
    df: pd.DataFrame,
    target_column: str = "Class"
) -> None:
    """
    Validate that the target column contains only 0 and 1.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_column : str
        Name of the target column.

    Raises
    ------
    ValueError
        If the target column is missing or contains invalid values.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    valid_values = {0, 1}
    actual_values = set(df[target_column].dropna().unique())

    invalid_values = actual_values - valid_values

    if invalid_values:
        raise ValueError(
            f"Invalid target values found: {invalid_values}"
        )


def validate_numeric_columns(
    df: pd.DataFrame,
    columns: list[str]
) -> None:
    """
    Validate that specified columns contain numeric data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list[str]
        Columns that must contain numeric values.

    Raises
    ------
    ValueError
        If a required column is missing or contains non-numeric data.
    """

    missing_columns = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    non_numeric_columns = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric_columns:
        raise ValueError(
            f"Non-numeric columns found: {non_numeric_columns}"
        )

def validate_model_input(
    df: pd.DataFrame,
    expected_columns: list[str]
) -> None:
    """
    Validate that input contains exactly the expected model features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    expected_columns : list[str]
        Exact feature columns expected by the model.

    Raises
    ------
    ValueError
        If columns are missing or unexpected columns are present.
    """

    actual_columns = set(df.columns)
    expected_columns_set = set(expected_columns)

    missing_columns = expected_columns_set - actual_columns
    unexpected_columns = actual_columns - expected_columns_set

    if missing_columns:
        raise ValueError(
            f"Missing model input columns: {sorted(missing_columns)}"
        )

    if unexpected_columns:
        raise ValueError(
            f"Unexpected model input columns: {sorted(unexpected_columns)}"
        )