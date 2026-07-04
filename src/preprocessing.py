"""
=========================================================
HeartWise Data Preprocessing
---------------------------------------------------------
Preprocessing pipeline for Heart Disease Risk Prediction

Author  : Kelompok ...
Version : 1.0
=========================================================
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    TARGET_COLUMN,
    DROP_COLUMNS,
    RANDOM_STATE,

    TRAIN_RAW_FILE,
    VALID_RAW_FILE,
    TEST_RAW_FILE,

    TRAIN_SCALED_FILE,
    VALID_SCALED_FILE,
    TEST_SCALED_FILE,

    SCALER_FILE,
    FEATURE_FILE
)

from src.utils import (
    log,
    save_pickle
)


# =========================================================
# REMOVE DUPLICATE
# =========================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows.
    """

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    log(f"Duplicate removed : {before-after}")
    log(f"Current rows      : {after}")

    return df


# =========================================================
# HANDLE MISSING VALUE
# =========================================================

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove missing values.
    """

    missing = df.isnull().sum().sum()

    log(f"Missing values : {missing}")

    before = len(df)

    df = df.dropna()

    after = len(df)

    log(f"Rows removed : {before-after}")

    return df

# =========================================================
# ENCODE CATEGORICAL
# =========================================================

def encode_categorical(df):
    """
    Encode categorical features.
    """

    categorical_columns = df.select_dtypes(include=["object", "string"]).columns

    for col in categorical_columns:

        if col.lower() == "sex":

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({
                    "male": 1,
                    "female": 0
                })
            )

        else:

            df[col] = (
                df[col]
                .astype("category")
                .cat.codes
            )

    log("Categorical features encoded.")

    return df

# =========================================================
# SPLIT FEATURE TARGET
# =========================================================

def split_feature_target(df):
    """
    Split dataframe into feature and target.
    """

    X = df.drop(
        columns=[TARGET_COLUMN] + DROP_COLUMNS
    )

    y = df[TARGET_COLUMN]

    log(f"Feature shape : {X.shape}")
    log(f"Target shape  : {y.shape}")

    return X, y

# =========================================================
# TRAIN VALIDATION TEST SPLIT
# =========================================================

def split_dataset(X, y):
    """
    Train : 70%
    Validation : 15%
    Test : 15%
    """

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    log("Dataset split completed.")

    log(f"Train      : {len(X_train)}")
    log(f"Validation : {len(X_valid)}")
    log(f"Test       : {len(X_test)}")

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test
    )


# =========================================================
# STANDARD SCALER
# =========================================================

def scaling(
    X_train,
    X_valid,
    X_test
):
    """
    Scale features using StandardScaler.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_valid_scaled = scaler.transform(X_valid)

    X_test_scaled = scaler.transform(X_test)

    save_pickle(
        scaler,
        SCALER_FILE
    )

    log("Scaler saved successfully.")

    return (
        X_train_scaled,
        X_valid_scaled,
        X_test_scaled
    )


# =========================================================
# SAVE FEATURE COLUMN
# =========================================================

def save_feature_columns(columns):
    """
    Save feature names.
    """

    save_pickle(
        list(columns),
        FEATURE_FILE
    )

    log("Feature columns saved.")

# =========================================================
# SAVE DATASET
# =========================================================

def save_dataset(
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
    y_test,
    X_train_scaled,
    X_valid_scaled,
    X_test_scaled
):
    """
    Save raw and scaled datasets.
    """

    # =====================================================
    # RAW DATASET
    # =====================================================

    train_raw = X_train.copy()
    train_raw[TARGET_COLUMN] = y_train.values

    valid_raw = X_valid.copy()
    valid_raw[TARGET_COLUMN] = y_valid.values

    test_raw = X_test.copy()
    test_raw[TARGET_COLUMN] = y_test.values

    train_raw.to_csv(
        TRAIN_RAW_FILE,
        index=False
    )

    valid_raw.to_csv(
        VALID_RAW_FILE,
        index=False
    )

    test_raw.to_csv(
        TEST_RAW_FILE,
        index=False
    )

    # =====================================================
    # SCALED DATASET
    # =====================================================

    train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns
    )

    train_scaled[TARGET_COLUMN] = y_train.values

    valid_scaled = pd.DataFrame(
        X_valid_scaled,
        columns=X_valid.columns
    )

    valid_scaled[TARGET_COLUMN] = y_valid.values

    test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns
    )

    test_scaled[TARGET_COLUMN] = y_test.values

    train_scaled.to_csv(
        TRAIN_SCALED_FILE,
        index=False
    )

    valid_scaled.to_csv(
        VALID_SCALED_FILE,
        index=False
    )

    test_scaled.to_csv(
        TEST_SCALED_FILE,
        index=False
    )

    log("Processed datasets saved successfully.")

    return (
        train_raw,
        valid_raw,
        test_raw,
        train_scaled,
        valid_scaled,
        test_scaled
    )

# =========================================================
# PREPROCESSING PIPELINE
# =========================================================

def preprocess_pipeline(df):
    """
    Complete preprocessing pipeline.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataset.

    Returns
    -------
    tuple
        Processed datasets.
    """

    log("=" * 60)
    log("START PREPROCESSING")
    log("=" * 60)

    # -----------------------------------------------------
    # Remove duplicate
    # -----------------------------------------------------

    df = remove_duplicates(df)

    # -----------------------------------------------------
    # Handle missing value
    # -----------------------------------------------------

    df = handle_missing(df)

    # -----------------------------------------------------
    # Encode categorical feature
    # -----------------------------------------------------

    df = encode_categorical(df)

    # -----------------------------------------------------
    # Split feature target
    # -----------------------------------------------------

    X, y = split_feature_target(df)

    # -----------------------------------------------------
    # Save feature columns
    # -----------------------------------------------------

    save_feature_columns(
        X.columns
    )

    # -----------------------------------------------------
    # Train Validation Test Split
    # -----------------------------------------------------

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test
    ) = split_dataset(
        X,
        y
    )

    # -----------------------------------------------------
    # Scaling
    # -----------------------------------------------------

    (
        X_train_scaled,
        X_valid_scaled,
        X_test_scaled
    ) = scaling(
        X_train,
        X_valid,
        X_test
    )

    # -----------------------------------------------------
    # Save Dataset
    # -----------------------------------------------------

    (
        train_raw,
        valid_raw,
        test_raw,
        train_scaled,
        valid_scaled,
        test_scaled
    ) = save_dataset(

        X_train,
        X_valid,
        X_test,

        y_train,
        y_valid,
        y_test,

        X_train_scaled,
        X_valid_scaled,
        X_test_scaled

    )

    log("=" * 60)
    log("PREPROCESSING FINISHED")
    log("=" * 60)

    return {

    # ===========================
    # RAW DATASET
    # ===========================

    "train_raw": train_raw,
    "valid_raw": valid_raw,
    "test_raw": test_raw,

    # ===========================
    # SCALED DATASET
    # ===========================

    "train_scaled": train_scaled,
    "valid_scaled": valid_scaled,
    "test_scaled": test_scaled,

    # ===========================
    # FEATURE & TARGET
    # ===========================

    "X_train": X_train_scaled,
    "X_valid": X_valid_scaled,
    "X_test": X_test_scaled,

    "y_train": y_train,
    "y_valid": y_valid,
    "y_test": y_test,

    # ===========================
    # FEATURE NAME
    # ===========================

    "feature_columns": list(X.columns)

}
