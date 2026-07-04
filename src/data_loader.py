"""
=========================================================
HeartWise Data Loader
---------------------------------------------------------
Load dataset and display basic dataset information.
=========================================================
"""

import pandas as pd

from src.config import DATASET_FILE
from src.utils import log, print_section


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(filepath=DATASET_FILE):
    """
    Load dataset.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    DataFrame
    """

    log(f"Loading dataset : {filepath}")

    df = pd.read_csv(filepath)

    log("Dataset loaded successfully.")

    return df


# =========================================================
# DATASET SHAPE
# =========================================================

def dataset_shape(df):
    """
    Display dataset shape.
    """

    print_section("DATASET SHAPE")

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")


# =========================================================
# DATASET INFO
# =========================================================

def dataset_info(df):
    """
    Display dataframe information.
    """

    print_section("DATASET INFO")

    df.info()


# =========================================================
# COLUMN LIST
# =========================================================

def column_list(df):
    """
    Display all columns.
    """

    print_section("COLUMN LIST")

    for i, col in enumerate(df.columns):

        print(f"{i+1}. {col}")


# =========================================================
# DATA TYPES
# =========================================================

def data_types(df):
    """
    Display data types.
    """

    print_section("DATA TYPES")

    print(df.dtypes)


# =========================================================
# DESCRIPTIVE STATISTICS
# =========================================================

def descriptive_statistics(df):
    """
    Display descriptive statistics.
    """

    print_section("DESCRIPTIVE STATISTICS")

    return df.describe().T


# =========================================================
# FIRST ROW
# =========================================================

def head(df, n=5):
    """
    Show first n rows.
    """

    print_section("FIRST ROW")

    return df.head(n)


# =========================================================
# LAST ROW
# =========================================================

def tail(df, n=5):
    """
    Show last n rows.
    """

    print_section("LAST ROW")

    return df.tail(n)


# =========================================================
# MISSING VALUE
# =========================================================

def missing_value(df):
    """
    Display missing values.
    """

    print_section("MISSING VALUE")

    missing = pd.DataFrame({

        "Missing": df.isnull().sum(),

        "Percentage (%)":
        round(df.isnull().mean()*100,2)

    })

    return missing


# =========================================================
# DUPLICATE
# =========================================================

def duplicate(df):
    """
    Display duplicate records.
    """

    print_section("DUPLICATE")

    print(f"Duplicate Data : {df.duplicated().sum()}")


# =========================================================
# NUMERICAL FEATURES
# =========================================================

def numerical_columns(df):
    """
    Return numerical columns.
    """

    return df.select_dtypes(include="number").columns.tolist()


# =========================================================
# CATEGORICAL FEATURES
# =========================================================

def categorical_columns(df):
    """
    Return categorical columns.
    """

    return df.select_dtypes(
        exclude="number"
    ).columns.tolist()


# =========================================================
# DATASET SUMMARY
# =========================================================

def dataset_summary(df):
    """
    Display complete dataset summary.
    """

    dataset_shape(df)

    dataset_info(df)

    column_list(df)

    data_types(df)

    duplicate(df)

    missing_value(df)

    return descriptive_statistics(df)